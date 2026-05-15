#!/usr/bin/env python3
"""
AsyncPrefetchLoader - Non-blocking data loader for DeepSDF training.

This module provides a producer-consumer pattern where background processes
continuously load batches into a queue, ensuring the GPU never waits for I/O.

Supports multiple producer processes for improved throughput and fault tolerance.

Thread-safety guarantees:
- Uses multiprocessing.Queue which is process-safe
- Multiple producer, single consumer pattern
- Stop event + timeout-based polling for shutdown
- Auto-restart with exponential backoff for resilience

Multiprocessing context rules (CRITICAL for correctness):
    All multiprocessing primitives — Process, Queue, Event — and the inner
    DataLoader's workers MUST use the same multiprocessing context.  Mixing
    contexts (e.g. spawn Process with fork DataLoader workers) causes
    deadlocks because fork copies locks but not the threads that hold them.

    This module enforces the rule by passing the caller-supplied ``mp_context``
    to every primitive, including the inner DataLoader via
    ``multiprocessing_context=``.

    On Linux the default start method is ``fork``.  If your program creates
    threads (PyTorch, NumPy, gRPC, etc.) before starting the loader, consider
    using ``mp.get_context("spawn")`` to avoid fork-safety hazards.

Platform safety:
    - **Linux + fork** (default):  Fast, but unsafe if any background threads
      exist when the producer process is forked.  Python 3.12+ emits a
      ``DeprecationWarning`` when ``os.fork()`` is called in a threaded process.
    - **Linux/macOS + spawn**:  Safe everywhere.  Slightly slower startup, but
      avoids all fork-safety issues.  **Recommended for production.**
    - **Windows**:  Only ``spawn`` is supported (``fork`` is unavailable).

Python 3.12+ resource_tracker workaround:
    When using ``spawn``, Python starts a ``resource_tracker`` process to clean
    up leaked semaphores / shared memory.  At interpreter shutdown the main
    process calls ``ResourceTracker.__del__`` → ``_stop`` → ``waitpid()``.
    If the tracker is still cleaning up, ``waitpid`` blocks forever — a known
    upstream regression (cpython#140485, pytorch#153050).  This module removes
    the problematic finalizer at import time so the OS reaps the tracker
    asynchronously.
"""

import time
import random
import logging
import multiprocessing as mp
import signal
from queue import Empty, Full
from typing import List, Dict, Any, Optional

# ---------------------------------------------------------------------------
# Python 3.12+ resource_tracker shutdown hang workaround.
#
# When using spawn, Python starts a resource_tracker process that tracks
# named semaphores / shared memory.  At interpreter shutdown the main
# process calls ResourceTracker.__del__ → _stop → waitpid(tracker_pid).
# If the tracker is still cleaning up leaked semaphores (e.g. from a
# Queue or PyTorch tensor sharing), waitpid blocks forever and the
# program hangs.
#
# Removing the finalizer lets the tracker exit asynchronously; the OS
# reaps it normally.
#   https://github.com/python/cpython/issues/140485
#   https://github.com/pytorch/pytorch/issues/153050
# ---------------------------------------------------------------------------
try:
    import multiprocessing.resource_tracker as _rt

    if hasattr(_rt.ResourceTracker, "__del__"):
        del _rt.ResourceTracker.__del__
except Exception:
    pass


class AsyncPrefetchLoader:
    """
    Non-blocking data loader that pre-fetches batches into a queue.

    One or more background processes continuously load data from disk and fill
    the queue, ensuring the training loop never waits for I/O.

    Multiple Producers Architecture:
        Producer 1 ─── DataLoader (N workers) ──┐
        Producer 2 ─── DataLoader (N workers) ──┼── Queue ─── GPU
        Producer 3 ─── DataLoader (N workers) ──┤
        Producer 4 ─── DataLoader (N workers) ──┘

    Context safety:
        The ``mp_context`` parameter controls the start method for ALL
        multiprocessing primitives — producers, Queue, Event, and the inner
        DataLoader's workers.  Using the same context everywhere prevents
        the fork-inside-spawn deadlock.

        Recommended: ``mp.get_context("spawn")`` for cross-platform safety.
        Default: ``mp.get_context()`` (fork on Linux, spawn on Windows/macOS).

    Usage:
        import multiprocessing as mp

        loader = AsyncPrefetchLoader(
            dataset, batch_size=32,
            num_producers=4, workers_per_producer=4,
            mp_context=mp.get_context("spawn"),  # recommended
            auto_restart=True,
        )
        loader.start()

        for epoch in range(num_epochs):
            for batch in loader:
                pass

        loader.stop()
    """

    def __init__(
        self,
        dataset,
        batch_size,
        num_workers: int = 8,
        num_producers: int = 1,
        workers_per_producer: int = None,
        queue_size: int = 32,
        seed: int = None,
        mp_context=None,
        prefetch_factor: int = 4,
        auto_restart: bool = True,
        max_restart_attempts: int = 3,
        restart_backoff_base: float = 1.0,
        max_epoch_retries: int = 3,
    ):
        """
        Initialize the async prefetch loader.

        Args:
            dataset: PyTorch Dataset to load from
            batch_size: Number of samples per batch
            num_workers: Legacy: total workers (used to derive workers_per_producer)
            num_producers: Number of producer processes (default: 1, single producer)
            workers_per_producer: Workers per producer (derived from num_workers if None)
            queue_size: Maximum number of batches to pre-load (buffer size)
            seed: Random seed for reproducibility
            mp_context: Multiprocessing context (e.g., mp.get_context("spawn"))
            prefetch_factor: DataLoader prefetch factor (default: 4)
            auto_restart: Auto-restart failed producers (default: True)
            max_restart_attempts: Max restart attempts per producer (default: 3)
            restart_backoff_base: Base seconds for exponential backoff (default: 1.0)
            max_epoch_retries: Max retries for incomplete epochs (default: 3)
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.queue_size = queue_size
        self.seed = seed if seed is not None else random.randint(0, 2**32 - 1)

        # Derive workers_per_producer if not specified
        if workers_per_producer is None:
            workers_per_producer = max(0, num_workers // num_producers)

        self.num_producers = num_producers
        self.workers_per_producer = workers_per_producer
        self.prefetch_factor = prefetch_factor
        self.auto_restart = auto_restart
        self.max_restart_attempts = max_restart_attempts
        self.restart_backoff_base = restart_backoff_base

        self._mp_ctx = mp_context or mp.get_context()

        # Use multiprocessing primitives for safe inter-process communication
        self.data_queue = self._mp_ctx.Queue(maxsize=queue_size)
        self.stop_event = self._mp_ctx.Event()

        # Simple state tracking (no dataclass)
        self._producer_processes: List[mp.Process] = []
        self._producer_seeds: List[int] = []  # seed for each producer
        self._restart_counts: List[int] = []  # restart count per producer
        self._last_restart_time: List[float] = []  # last restart timestamp per producer

        self._batches_per_epoch = len(dataset) // batch_size
        self._current_epoch: int = 0  # Tracks training epoch for producer restart coordination
        self._stopped = False  # Guard against multiple stop() calls

        # Epoch retry and failure tracking
        self.max_epoch_retries = max_epoch_retries
        self._epoch_retries: int = 0  # Current epoch retry count
        self._failure_log: List[Dict[str, Any]] = []  # Records of producer failures

    def _drain_queue(self, limit=None):
        """Best-effort drain of the queue without relying on .empty()."""
        drained = 0
        while True:
            if limit is not None and drained >= limit:
                return drained
            try:
                self.data_queue.get_nowait()
                drained += 1
            except Empty:
                return drained
            except Exception:
                return drained

    @staticmethod
    def _producer_loop(
        dataset,
        batch_size,
        num_workers,
        data_queue,
        stop_event,
        seed,
        producer_id: int = 0,
        num_producers: int = 1,
        epoch_base_seed: int = 0,
        start_epoch: int = 0,
        prefetch_factor: int = 4,
        mp_ctx=None,
        shared_epoch=None,
    ):
        """
        Background process that loads batches into the queue.

        This runs in a separate process and continuously:
        1. Checks if stopped
        2. Loads a deterministic strided partition of the epoch's batches
        3. Puts batches into the queue for consumption

        All producers share the same epoch shuffle (via epoch_base_seed) but
        each takes a strided slice of the resulting batch list so that together
        they cover exactly one full epoch without overlap or duplication.

        Epoch fencing:
            ``shared_epoch`` is a ``multiprocessing.Value('i')`` controlled by the
            consumer.  A producer will NEVER produce an epoch more than 1 ahead
            of ``shared_epoch.value``.  This prevents the fast-producer race that
            otherwise causes batches from epoch N+2 (and beyond) to leak into
            the consumer's current epoch.

        start_epoch: epoch number to resume from (used when a producer is
        restarted mid-training so its shuffle aligns with the surviving producers).
        """
        import torch
        import random
        import numpy as np
        from torch.utils.data import DataLoader, Subset
        # Let the parent process handle Ctrl+C and orchestrate shutdown.
        # This avoids noisy KeyboardInterrupt traces from child processes.
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        # When sending tensors across an mp.Queue, PyTorch may use shared-memory
        # file descriptors by default, which can fail on some systems / limits.
        # Fall back to filesystem-based sharing which is generally more robust.
        try:
            import torch.multiprocessing as torch_mp

            torch_mp.set_sharing_strategy("file_system")
        except Exception:
            pass

        # Set seeds for reproducibility in this process
        # Each producer uses unique seed for independent shuffling
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        current_epoch = start_epoch
        producer_tag = f"[AsyncLoader-P{producer_id}]"

        while not stop_event.is_set():
            # --- epoch fence ---------------------------------------------------
            # NEVER produce batches for an epoch newer than what the consumer
            # is currently training on.  Even +1 ahead causes leakage: a fast
            # producer can finish its partition and start pushing epoch N+1
            # batches into the queue while the consumer is still on epoch N.
            if shared_epoch is not None:
                if current_epoch > shared_epoch.value:
                    time.sleep(0.1)
                    continue
            # ------------------------------------------------------------------

            try:
                # All producers use the same deterministic epoch shuffle so that
                # their strided partitions together cover the full dataset exactly once.
                epoch_shuffle_seed = epoch_base_seed + current_epoch * 99991
                epoch_rng = random.Random(epoch_shuffle_seed)
                indices = list(range(len(dataset)))
                epoch_rng.shuffle(indices)

                # Partition batches across producers using a stride pattern.
                # Producer i handles batch slots [i, i+N, i+2N, ...] so that
                # all N producers together cover exactly one full epoch with no overlap.
                total_batches = len(indices) // batch_size
                valid_indices = []
                for b in range(producer_id, total_batches, num_producers):
                    valid_indices.extend(indices[b * batch_size:(b + 1) * batch_size])

                if not valid_indices:
                    # This producer has no batches assigned this epoch — happens when
                    # num_producers > total_batches. Sleep briefly to avoid a tight spin.
                    logging.warning(
                        f"{producer_tag} No batches assigned for epoch {current_epoch} "
                        f"(dataset too small for {num_producers} producers). Skipping."
                    )
                    time.sleep(1.0)
                    current_epoch += 1
                    continue

                subset = Subset(dataset, valid_indices)
                loader_kwargs = dict(
                    batch_size=batch_size,
                    shuffle=False,  # Already shuffled
                    num_workers=num_workers,
                    drop_last=True,
                    pin_memory=False,  # Queue IPC breaks pinning benefit; consumer uses non_blocking instead
                    prefetch_factor=prefetch_factor if num_workers > 0 else None,
                )
                # Use the same multiprocessing context for DataLoader workers as the
                # producer process itself. This prevents fork-after-spawn deadlocks
                # when the loader is run under spawn-based execution (e.g. conda run).
                if num_workers > 0 and mp_ctx is not None:
                    loader_kwargs["multiprocessing_context"] = mp_ctx
                epoch_loader = DataLoader(subset, **loader_kwargs)

                logging.debug(f"{producer_tag} Started pre-loading epoch {current_epoch}")

                batch_count = 0
                for batch in epoch_loader:
                    if stop_event.is_set():
                        logging.info(f"{producer_tag} Stop signal received, exiting")
                        return

                    sdf_data, indices_tensor = batch

                    # Clone tensors to ensure we own the memory before sharing
                    # This avoids issues with DataLoader's memory management
                    sdf_data = sdf_data.clone()
                    indices_tensor = indices_tensor.clone()

                    # Put with timeout to allow checking stop_event
                    put_start = time.time()
                    while not stop_event.is_set():
                        try:
                            data_queue.put((current_epoch, sdf_data, indices_tensor), timeout=0.5)
                            put_time = time.time() - put_start
                            batch_count += 1
                            if batch_count % 10 == 0:
                                logging.debug(
                                    f"{producer_tag} put batch {batch_count} (waited {put_time:.3f}s)"
                                )
                            break
                        except Full:
                            continue  # Queue full, wait and retry

                current_epoch += 1
                logging.debug(f"{producer_tag} Epoch {current_epoch - 1} complete")

            except Exception as e:
                if not stop_event.is_set():
                    logging.exception(f"{producer_tag} Producer error")
                    time.sleep(0.5)

        logging.info(f"{producer_tag} Producer stopped")

    def _check_producer_health(self) -> List[int]:
        """
        Return list of failed producer IDs (indices into _producer_processes).

        A producer is considered failed if:
        - The process is None or not alive
        """
        failed = []
        for i, process in enumerate(self._producer_processes):
            if process is None or not process.is_alive():
                failed.append(i)
        return failed

    def _restart_failed_producers(self) -> None:
        """Restart failed producers with exponential backoff."""
        if not self.auto_restart:
            return

        failed_ids = self._check_producer_health()
        current_time = time.time()

        for pid in failed_ids:
            # Check backoff timing
            time_since_last = current_time - self._last_restart_time[pid]
            backoff_time = self.restart_backoff_base * (2 ** self._restart_counts[pid])

            if time_since_last < backoff_time:
                continue  # Still in backoff period

            # Check restart limit
            if self._restart_counts[pid] >= self.max_restart_attempts:
                logging.warning(
                    f"[AsyncLoader] Producer {pid} exceeded max restart attempts "
                    f"({self.max_restart_attempts}), not restarting"
                )
                continue

            # Restart the producer
            logging.warning(
                f"[AsyncLoader] Restarting failed producer {pid} "
                f"(attempt {self._restart_counts[pid] + 1}/{self.max_restart_attempts})"
            )

            # Create new process
            process = self._mp_ctx.Process(
                target=self._producer_loop,
                args=(
                    self.dataset,
                    self.batch_size,
                    self.workers_per_producer,
                    self.data_queue,
                    self.stop_event,
                    self._producer_seeds[pid],
                    pid,  # producer_id
                    self.num_producers,
                    self.seed,  # epoch_base_seed: shared across all producers
                    self._current_epoch,  # start_epoch: resume from current training epoch
                    self.prefetch_factor,
                    self._mp_ctx,  # pass context so DataLoader workers match
                    self._shared_epoch,  # epoch fence
                ),
                daemon=False,
            )
            process.start()

            # Update state
            self._producer_processes[pid] = process
            self._restart_counts[pid] += 1
            self._last_restart_time[pid] = current_time

    def get_producer_status(self) -> Dict[str, Any]:
        """
        Get status dict with producer health information.

        Returns:
            Dict with keys:
            - 'running': number of running producers
            - 'total': total number of producers
            - 'queue_fill': current queue fill level
            - 'restarts': list of restart counts per producer
        """
        running = sum(
            1 for p in self._producer_processes
            if p is not None and p.is_alive()
        )
        return {
            'running': running,
            'total': self.num_producers,
            'queue_fill': self.get_queue_fill(),
            'restarts': list(self._restart_counts),
        }

    def _record_failure(self, producer_id: int, epoch: int, reason: str, details: Any = None):
        """Record a producer failure for training quality tracking."""
        entry = {
            'timestamp': time.time(),
            'producer_id': producer_id,
            'epoch': epoch,
            'reason': reason,
            'details': details,
        }
        self._failure_log.append(entry)
        logging.warning(
            f"[AsyncLoader] Failure recorded: producer={producer_id}, epoch={epoch}, "
            f"reason={reason}, details={details}"
        )

    def get_failure_stats(self) -> Dict[str, Any]:
        """
        Get failure statistics for training quality assessment.

        Returns:
            Dict with keys:
            - 'total_failures': total number of recorded failures
            - 'epoch_retries': number of epoch retries that occurred
            - 'failures_by_reason': count of failures grouped by reason
            - 'failures_by_epoch': count of failures grouped by epoch
            - 'failure_log': list of all failure records
        """
        by_reason = {}
        by_epoch = {}
        for entry in self._failure_log:
            reason = entry['reason']
            epoch = entry['epoch']
            by_reason[reason] = by_reason.get(reason, 0) + 1
            by_epoch[epoch] = by_epoch.get(epoch, 0) + 1

        return {
            'total_failures': len(self._failure_log),
            'epoch_retries': self._epoch_retries,
            'failures_by_reason': by_reason,
            'failures_by_epoch': by_epoch,
            'failure_log': list(self._failure_log),
        }

    def start(self):
        """Start the background producer processes."""
        # Check if any producers are already running
        if any(p is not None and p.is_alive() for p in self._producer_processes):
            logging.warning("[AsyncLoader] Some producers already running")
            return

        # Reset stopped flag for restart
        self._stopped = False

        # Match producer sharing strategy for robust cross-process tensor IPC.
        try:
            import torch.multiprocessing as torch_mp

            torch_mp.set_sharing_strategy("file_system")
        except Exception:
            pass

        # Clear any stale data from old queue before replacing it.
        self._drain_queue()

        # Close old resources to prevent leaks (shared memory / semaphores left
        # behind confuse the resource_tracker and can cause shutdown hangs).
        if hasattr(self, "data_queue") and self.data_queue is not None:
            try:
                if hasattr(self.data_queue, "cancel_join_thread"):
                    self.data_queue.cancel_join_thread()
                self.data_queue.close()
            except Exception:
                pass

        # Fresh primitives for each run (allows clean restart after stop()).
        self.data_queue = self._mp_ctx.Queue(maxsize=self.queue_size)
        self.stop_event = self._mp_ctx.Event()
        self._shared_epoch = self._mp_ctx.Value('i', 0)  # epoch fence

        # Initialize state tracking lists
        self._producer_processes = []
        self._producer_seeds = [self.seed + i * 1000 for i in range(self.num_producers)]
        self._restart_counts = [0] * self.num_producers
        self._last_restart_time = [0.0] * self.num_producers
        # Recompute in case dataset changed since __init__ (e.g. stop/start cycle)
        self._batches_per_epoch = len(self.dataset) // self.batch_size
        self._current_epoch = 0  # Reset epoch counter on fresh start

        for i in range(self.num_producers):
            process = self._mp_ctx.Process(
                target=self._producer_loop,
                args=(
                    self.dataset,
                    self.batch_size,
                    self.workers_per_producer,
                    self.data_queue,
                    self.stop_event,
                    self._producer_seeds[i],
                    i,  # producer_id
                    self.num_producers,
                    self.seed,  # epoch_base_seed: shared across all producers
                    0,  # start_epoch: always 0 on fresh start
                    self.prefetch_factor,
                    self._mp_ctx,  # pass context so DataLoader workers match
                    self._shared_epoch,  # epoch fence: prevent race-ahead
                ),
                daemon=False,  # Must be False - producer spawns DataLoader workers
            )
            process.start()
            self._producer_processes.append(process)

        logging.info(
            f"[AsyncLoader] Started {self.num_producers} producers, "
            f"{self.workers_per_producer} workers each, queue size {self.queue_size}"
        )

    def stop(self, immediate=False):
        """Stop all background producer processes.

        Args:
            immediate: If True, terminate quickly (used for Ctrl+C paths)
                instead of waiting for graceful worker teardown.
        """
        # Idempotent: only stop once
        if self._stopped:
            return
        self._stopped = True

        self.stop_event.set()

        # Clear queue to reduce time spent blocked on a full queue
        self._drain_queue()

        # Stop all producer processes
        for i, process in enumerate(self._producer_processes):
            if process is None:
                continue

            if immediate:
                if process.is_alive():
                    # Try SIGTERM first
                    process.terminate()
                    process.join(timeout=0.5)
                    # Force kill if still alive
                    if process.is_alive():
                        try:
                            process.kill()
                        except Exception:
                            pass
                    # Don't wait for join - just move on to prevent hanging
                    # The OS will reap the zombie process
            else:
                process.join(timeout=3)
                if process.is_alive():
                    process.terminate()
                    # Short timeout, don't hang waiting
                    process.join(timeout=1)

        # Clear processes list
        self._producer_processes = []

        # Clean up Queue resources so interpreter shutdown doesn't hang.
        # In immediate mode, skip close() to avoid blocking on the feeder thread.
        if not immediate:
            try:
                if hasattr(self.data_queue, "cancel_join_thread"):
                    self.data_queue.cancel_join_thread()
                self.data_queue.close()
            except Exception:
                pass
        else:
            # For immediate stop, just cancel join thread without close()
            try:
                if hasattr(self.data_queue, "cancel_join_thread"):
                    self.data_queue.cancel_join_thread()
            except Exception:
                pass

        logging.info("[AsyncLoader] Stopped")

    def get_queue_fill(self):
        """
        Return current number of batches in the queue.

        Note: On some platforms this may be approximate or raise NotImplementedError.
        Use for monitoring only, not for control flow.
        """
        try:
            return self.data_queue.qsize()
        except NotImplementedError:
            return -1  # Not supported on this platform
        except Exception:
            return -1

    def flush_epoch(self):
        """
        Epoch barrier: drain the queue and advance the shared epoch counter.

        Call this **between training epochs** to prevent batch leakage across
        epoch boundaries.  Producers are fenced to at most 1 epoch ahead, and
        this drain removes any prefetched batches that bled through.

        Without this call, fast producers can race ahead to epoch N+2 (and
        beyond) before the consumer finishes epoch N, mixing batches from
        different epochs in the queue.
        """
        self._drain_queue()
        if self._shared_epoch is not None:
            self._shared_epoch.value = self._current_epoch

    def __iter__(self):
        """
        Iterate over up to one epoch's worth of batches.

        Normally yields exactly _batches_per_epoch batches and then increments
        the internal epoch counter. If producers die mid-epoch, the epoch is
        retried up to max_epoch_retries times with queue drain between attempts.

        Features:
        - Health monitoring: checks that at least one producer is running
        - Auto-restart: periodically restarts failed producers if enabled
        - Epoch retry: retries incomplete epochs with failure recording
        - Graceful exit: exits only if ALL producers are dead or max retries exceeded

        Note: We intentionally do not send epoch sentinel markers through the
        multiprocessing queue. Sentinel objects are brittle across processes,
        and prefetching multiple epochs can leave stale epoch markers that
        would cause later epochs to terminate early.
        """
        self._epoch_retries = 0

        while True:  # Epoch retry loop
            batches_yielded = 0
            batch_count = 0  # For periodic health check
            epoch_start_time = time.time()
            failed_producers_this_epoch = set()

            # Check that at least one producer is running
            running_count = sum(
                1 for p in self._producer_processes
                if p is not None and p.is_alive()
            )
            if running_count == 0:
                logging.error("[AsyncLoader] No producers running; did you call start()?")
                return

            while batches_yielded < self._batches_per_epoch:
                if self.stop_event.is_set():
                    return

                # Periodic health check (every 100 batches)
                if self.auto_restart and batch_count > 0 and batch_count % 100 == 0:
                    failed_ids = self._check_producer_health()
                    for fid in failed_ids:
                        if fid not in failed_producers_this_epoch:
                            failed_producers_this_epoch.add(fid)
                            self._record_failure(fid, self._current_epoch, "detected_at_batch", batch_count)
                    self._restart_failed_producers()

                    # Check if all producers are dead
                    running_count = sum(
                        1 for p in self._producer_processes
                        if p is not None and p.is_alive()
                    )
                    if running_count == 0:
                        logging.warning(
                            f"[AsyncLoader] All producers died at epoch {self._current_epoch}, "
                            f"batch {batches_yielded}/{self._batches_per_epoch} — epoch truncated"
                        )
                        break  # Exit inner loop, will retry or give up

                try:
                    # Get with timeout to allow checking stop_event
                    get_start = time.time()
                    item = self.data_queue.get(timeout=2.0)
                    get_time = time.time() - get_start

                    # --- epoch-tag gate ---
                    # Discard batches from wrong epochs.  The fence prevents
                    # this in normal operation, but this gate is defense-in-depth
                    # against any remaining races or bugs.
                    epoch_tag, sdf_data, indices = item
                    if epoch_tag != self._current_epoch:
                        logging.debug(
                            "[AsyncLoader] Discarded epoch-%d batch (current=%d)",
                            epoch_tag, self._current_epoch,
                        )
                        continue  # don't count toward _batches_per_epoch
                    # -----------------------

                    batches_yielded += 1
                    batch_count += 1

                    queue_fill = self.get_queue_fill()
                    if batches_yielded % 10 == 0 or get_time > 0.1:
                        logging.debug(
                            f"[AsyncLoader-Consumer] batch {batches_yielded}/{self._batches_per_epoch}, "
                            f"queue={queue_fill}, get_time={get_time:.3f}s"
                        )

                    yield sdf_data, indices

                except Empty:
                    # Timeout - check if at least one producer is still alive
                    running_count = sum(
                        1 for p in self._producer_processes
                        if p is not None and p.is_alive()
                    )
                    if running_count == 0:
                        logging.warning(
                            f"[AsyncLoader] All producers died unexpectedly at epoch "
                            f"{self._current_epoch}, batch {batches_yielded}/{self._batches_per_epoch} "
                            f"— epoch truncated"
                        )
                        break  # Exit inner loop, will retry or give up

                    continue  # Keep waiting

            # Check if epoch completed successfully
            epoch_elapsed = time.time() - epoch_start_time
            if batches_yielded >= self._batches_per_epoch:
                # Full epoch complete — advance the epoch counter
                self._current_epoch += 1
                if self._shared_epoch is not None:
                    self._shared_epoch.value = self._current_epoch

                # Record success if this was a retry
                if self._epoch_retries > 0:
                    logging.info(
                        f"[AsyncLoader] Epoch {self._current_epoch - 1} completed after "
                        f"{self._epoch_retries} retry(ies) ({epoch_elapsed:.1f}s)"
                    )
                self._epoch_retries = 0
                return  # Success — exit generator

            # Epoch incomplete — handle retry
            self._epoch_retries += 1
            if self._epoch_retries > self.max_epoch_retries:
                logging.error(
                    f"[AsyncLoader] Epoch {self._current_epoch} failed after "
                    f"{self.max_epoch_retries} retries — giving up"
                )
                self._record_failure(
                    -1, self._current_epoch, "max_retries_exceeded",
                    {"retries": self._epoch_retries, "batches_yielded": batches_yielded}
                )
                self._epoch_retries = 0
                return  # Give up — epoch is lost

            # Drain stale batches and retry
            drained = self._drain_queue()
            logging.warning(
                f"[AsyncLoader] Epoch {self._current_epoch} incomplete "
                f"({batches_yielded}/{self._batches_per_epoch} batches, "
                f"{len(failed_producers_this_epoch)} producer(s) failed) — "
                f"retry {self._epoch_retries}/{self.max_epoch_retries} "
                f"(drained {drained} stale batches)"
            )
            self._record_failure(
                -1, self._current_epoch, "epoch_incomplete",
                {
                    "batches_yielded": batches_yielded,
                    "batches_expected": self._batches_per_epoch,
                    "failed_producers": list(failed_producers_this_epoch),
                    "drained": drained,
                    "elapsed": epoch_elapsed,
                }
            )
            # Brief pause to let restarted producers fill the queue
            time.sleep(1.0)
            # Loop continues — retry same epoch

    def __len__(self):
        """Return number of batches per epoch."""
        return self._batches_per_epoch

    def _verify_batch_invariants(
        self,
        consumed_batch_ids: set,
        expected_scenes: set,
        epoch: int,
    ):
        """Verify scene-level invariants for an epoch.

        Each batch is identified by its frozenset of scene names.  Checks:
        1. Completeness — every expected scene appears in exactly one consumed batch
        2. Fidelity — no unexpected scenes consumed
        """
        consumed_scenes = set()
        for batch_id in consumed_batch_ids:
            consumed_scenes.update(batch_id)

        missing_scenes = expected_scenes - consumed_scenes
        extra_scenes = consumed_scenes - expected_scenes
        duplicate_count = len(consumed_batch_ids) - len(
            set(id(b) for b in consumed_batch_ids)
        )

        ok = True
        if missing_scenes:
            logging.warning(
                f"[AsyncLoader] Epoch {epoch}: {len(missing_scenes)} scenes MISSING "
                f"(consumed {len(consumed_scenes)}/{len(expected_scenes)}). "
                f"First missing: {sorted(missing_scenes)[:10]}"
            )
            ok = False
        if extra_scenes:
            logging.warning(
                f"[AsyncLoader] Epoch {epoch}: {len(extra_scenes)} UNEXPECTED scenes"
            )
            ok = False
        if ok:
            logging.info(
                f"[AsyncLoader] Epoch {epoch}: all {len(expected_scenes)} scenes "
                f"covered across {len(consumed_batch_ids)} batches"
            )

    def __del__(self):
        """Cleanup on deletion.

        NOTE: We intentionally do NOT call self.stop() here. During interpreter
        shutdown module globals may already be None, and stop() can hang
        indefinitely trying to join processes or close queues. Explicit cleanup
        is the caller's responsibility (train_deep_sdf.py calls stop() in its
        cleanup_resources() handler).
        """
        pass
