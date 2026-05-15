#!/usr/bin/env python3
"""
AsyncSpawnLoader — platform-agnostic replacement for AsyncPrefetchLoader.

Uses ``spawn`` everywhere (Linux, macOS, Windows) so behaviour is identical
regardless of launch command:

    python train.py
    conda run python train.py
    torchrun ...

This module reproduces all production features of AsyncPrefetchLoader:
- Multiple producer processes with strided epoch partitioning
- Auto-restart with exponential backoff
- Health monitoring and status reporting
- Per-epoch deterministic shuffle coordination
- Graceful and immediate shutdown paths
- Queue-fill telemetry

Design rules enforced:
- One multiprocessing context (spawn) for Process, Queue, Event, and DataLoader.
- No __del__ cleanup — use the context manager or call stop() explicitly.
- Old queues are closed before being replaced (no resource_tracker leaks).
- Queue feeder threads are detached before close (no join hangs).
- The inner DataLoader receives the same context as the producer process.
"""

import logging
import random
import signal
import time
import multiprocessing as mp
from queue import Empty, Full
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Module-level spawn context — hard-coded, never platform-dependent.
# ---------------------------------------------------------------------------
MP_CTX = mp.get_context("spawn")

# ---------------------------------------------------------------------------
# Workaround for Python 3.12+ resource_tracker hang at shutdown.
#
# When using spawn, Python starts a resource_tracker process that tracks
# named semaphores / shared memory.  At interpreter shutdown the main
# process calls ResourceTracker.__del__ → _stop → waitpid(tracker_pid).
# If the tracker is still cleaning up leaked semaphores (e.g. from a
# Queue or PyTorch tensor sharing), waitpid blocks forever and the
# program hangs.
#
# Removing the finalizer lets the tracker exit asynchronously; the OS
# reaps it normally.  This is a known upstream regression:
#   https://github.com/python/cpython/issues/140485
#   https://github.com/pytorch/pytorch/issues/153050
# ---------------------------------------------------------------------------
try:
    import multiprocessing.resource_tracker as _rt

    if hasattr(_rt.ResourceTracker, "__del__"):
        del _rt.ResourceTracker.__del__
except Exception:
    pass


def _safe_queue_close(q) -> None:
    """Close a multiprocessing.Queue without blocking on the feeder thread."""
    if q is None:
        return
    try:
        if hasattr(q, "cancel_join_thread"):
            q.cancel_join_thread()
    except Exception:
        pass
    try:
        q.close()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Producer loop (runs in a separate process)
# ---------------------------------------------------------------------------

def _producer_loop(
    dataset,
    batch_size: int,
    num_workers: int,
    data_queue: mp.Queue,
    stop_event: mp.Event,
    seed: int,
    producer_id: int,
    num_producers: int,
    epoch_base_seed: int,
    start_epoch: int,
    prefetch_factor: int,
    shared_epoch=None,
):
    """
    Background producer that loads a strided partition of each epoch.

    All producers share the same deterministic shuffle (via epoch_base_seed) but
    each takes a strided slice of the batch list so that together they cover
    exactly one full epoch without overlap or duplication.
    """
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Subset

    # Let the parent handle Ctrl+C.
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    # Robust tensor sharing across processes.
    try:
        import torch.multiprocessing as torch_mp

        torch_mp.set_sharing_strategy("file_system")
    except Exception:
        pass

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    current_epoch = start_epoch
    tag = f"[AsyncSpawnLoader-P{producer_id}]"

    while not stop_event.is_set():
        # --- epoch fence ---
        if shared_epoch is not None:
            if current_epoch > shared_epoch.value:
                time.sleep(0.1)
                continue
        # -------------------

        try:
            # Deterministic epoch shuffle shared by all producers.
            epoch_shuffle_seed = epoch_base_seed + current_epoch * 99991
            rng = random.Random(epoch_shuffle_seed)
            indices = list(range(len(dataset)))
            rng.shuffle(indices)

            # Strided partition: producer i handles batches i, i+N, i+2N, ...
            total_batches = len(indices) // batch_size
            valid_indices = []
            for b in range(producer_id, total_batches, num_producers):
                valid_indices.extend(indices[b * batch_size : (b + 1) * batch_size])

            if not valid_indices:
                logging.warning(
                    "%s No batches for epoch %d (dataset too small for %d producers).",
                    tag, current_epoch, num_producers,
                )
                time.sleep(1.0)
                current_epoch += 1
                continue

            subset = Subset(dataset, valid_indices)
            loader_kwargs = dict(
                batch_size=batch_size,
                shuffle=False,  # already shuffled above
                num_workers=num_workers,
                drop_last=True,
                pin_memory=False,
                prefetch_factor=prefetch_factor if num_workers > 0 else None,
            )
            # CRITICAL: pass spawn context so DataLoader workers match the producer.
            if num_workers > 0:
                loader_kwargs["multiprocessing_context"] = MP_CTX

            epoch_loader = DataLoader(subset, **loader_kwargs)
            logging.debug("%s Started epoch %d", tag, current_epoch)

            batch_count = 0
            for batch in epoch_loader:
                if stop_event.is_set():
                    logging.info("%s Stop signal received, exiting", tag)
                    return

                sdf_data, indices_tensor = batch
                sdf_data = sdf_data.clone()
                indices_tensor = indices_tensor.clone()

                put_start = time.time()
                while not stop_event.is_set():
                    try:
                        data_queue.put((current_epoch, sdf_data, indices_tensor), timeout=0.5)
                        put_time = time.time() - put_start
                        batch_count += 1
                        if batch_count % 10 == 0:
                            print(
                                f"[{time.strftime('%H:%M:%S')}] {tag} "
                                f"Put batch {batch_count} (waited {put_time:.3f}s)"
                            )
                        break
                    except Full:
                        continue

            current_epoch += 1
            logging.debug("%s Epoch %d complete", tag, current_epoch - 1)

        except Exception:
            if not stop_event.is_set():
                logging.exception("%s Producer error", tag)
                time.sleep(0.5)

    logging.info("%s Producer stopped", tag)


# ---------------------------------------------------------------------------
# Consumer / orchestrator
# ---------------------------------------------------------------------------

class AsyncSpawnLoader:
    """
    Drop-in platform-agnostic replacement for AsyncPrefetchLoader.

    Usage (recommended):
        loader = AsyncSpawnLoader(dataset, batch_size=16, num_producers=4)
        with loader:
            for epoch in range(num_epochs):
                for sdf_data, indices in loader:
                    ...

    Usage (explicit):
        loader.start()
        for batch in loader:
            ...
        loader.stop()
    """

    def __init__(
        self,
        dataset,
        batch_size: int,
        num_workers: int = 8,
        num_producers: int = 1,
        workers_per_producer: Optional[int] = None,
        queue_size: int = 32,
        seed: Optional[int] = None,
        prefetch_factor: int = 4,
        auto_restart: bool = True,
        max_restart_attempts: int = 3,
        restart_backoff_base: float = 1.0,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_producers = num_producers
        self.queue_size = queue_size
        self.seed = seed if seed is not None else random.randint(0, 2**32 - 1)
        self.prefetch_factor = prefetch_factor
        self.auto_restart = auto_restart
        self.max_restart_attempts = max_restart_attempts
        self.restart_backoff_base = restart_backoff_base

        if workers_per_producer is None:
            workers_per_producer = max(0, num_workers // num_producers)
        self.workers_per_producer = workers_per_producer

        # Runtime state
        self._batches_per_epoch = len(dataset) // batch_size
        self._current_epoch: int = 0
        self._stopped: bool = False

        self._data_queue: Optional[mp.Queue] = None
        self._stop_event: Optional[mp.Event] = None
        self._producer_processes: List[mp.Process] = []
        self._producer_seeds: List[int] = []
        self._restart_counts: List[int] = []
        self._last_restart_time: List[float] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _drain_queue(self, limit: Optional[int] = None) -> int:
        """Best-effort drain without relying on .empty()."""
        if self._data_queue is None:
            return 0
        drained = 0
        while True:
            if limit is not None and drained >= limit:
                return drained
            try:
                self._data_queue.get_nowait()
                drained += 1
            except Empty:
                return drained
            except Exception:
                return drained

    def _check_producer_health(self) -> List[int]:
        """Return indices of failed producers."""
        failed = []
        for i, proc in enumerate(self._producer_processes):
            if proc is None or not proc.is_alive():
                failed.append(i)
        return failed

    def _restart_failed_producers(self) -> None:
        """Restart dead producers with exponential backoff."""
        if not self.auto_restart:
            return

        failed_ids = self._check_producer_health()
        now = time.time()

        for pid in failed_ids:
            time_since_last = now - self._last_restart_time[pid]
            backoff = self.restart_backoff_base * (2 ** self._restart_counts[pid])
            if time_since_last < backoff:
                continue

            if self._restart_counts[pid] >= self.max_restart_attempts:
                logging.warning(
                    "[AsyncSpawnLoader] Producer %d exceeded max restarts (%d)",
                    pid, self.max_restart_attempts,
                )
                continue

            logging.warning(
                "[AsyncSpawnLoader] Restarting producer %d (attempt %d/%d)",
                pid, self._restart_counts[pid] + 1, self.max_restart_attempts,
            )

            proc = MP_CTX.Process(
                target=_producer_loop,
                args=(
                    self.dataset,
                    self.batch_size,
                    self.workers_per_producer,
                    self._data_queue,
                    self._stop_event,
                    self._producer_seeds[pid],
                    pid,
                    self.num_producers,
                    self.seed,
                    self._current_epoch,
                    self.prefetch_factor,
                    self._shared_epoch,  # epoch fence
                ),
                daemon=False,
            )
            proc.start()

            self._producer_processes[pid] = proc
            self._restart_counts[pid] += 1
            self._last_restart_time[pid] = now

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start all background producer processes."""
        if any(p is not None and p.is_alive() for p in self._producer_processes):
            logging.warning("[AsyncSpawnLoader] Some producers already running")
            return

        self._stopped = False

        # Close old queue if restarting (prevents resource_tracker leaks).
        if self._data_queue is not None:
            _safe_queue_close(self._data_queue)

        self._data_queue = MP_CTX.Queue(maxsize=self.queue_size)
        self._stop_event = MP_CTX.Event()
        self._shared_epoch = MP_CTX.Value('i', 0)  # epoch fence

        self._producer_processes = []
        self._producer_seeds = [self.seed + i * 1000 for i in range(self.num_producers)]
        self._restart_counts = [0] * self.num_producers
        self._last_restart_time = [0.0] * self.num_producers
        self._batches_per_epoch = len(self.dataset) // self.batch_size
        self._current_epoch = 0

        for i in range(self.num_producers):
            proc = MP_CTX.Process(
                target=_producer_loop,
                args=(
                    self.dataset,
                    self.batch_size,
                    self.workers_per_producer,
                    self._data_queue,
                    self._stop_event,
                    self._producer_seeds[i],
                    i,
                    self.num_producers,
                    self.seed,
                    0,
                    self.prefetch_factor,
                    self._shared_epoch,  # epoch fence
                ),
                daemon=False,
            )
            proc.start()
            self._producer_processes.append(proc)

        logging.info(
            "[AsyncSpawnLoader] Started %d producers, %d workers each, queue %d",
            self.num_producers, self.workers_per_producer, self.queue_size,
        )

    def stop(self, immediate: bool = False) -> None:
        """Stop all producers and release queue resources."""
        if self._stopped:
            return
        self._stopped = True

        if self._stop_event is not None:
            self._stop_event.set()

        self._drain_queue()

        for proc in self._producer_processes:
            if proc is None:
                continue
            if immediate:
                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=0.5)
                    if proc.is_alive():
                        try:
                            proc.kill()
                        except Exception:
                            pass
            else:
                proc.join(timeout=3)
                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=1)

        self._producer_processes = []

        if not immediate:
            _safe_queue_close(self._data_queue)
        else:
            # Immediate path: just cancel join thread, don't block on close().
            if self._data_queue is not None:
                try:
                    if hasattr(self._data_queue, "cancel_join_thread"):
                        self._data_queue.cancel_join_thread()
                except Exception:
                    pass

        self._data_queue = None
        self._stop_event = None
        logging.info("[AsyncSpawnLoader] Stopped")

    def get_producer_status(self) -> Dict[str, Any]:
        """Return producer health and queue-fill information."""
        running = sum(
            1 for p in self._producer_processes
            if p is not None and p.is_alive()
        )
        return {
            "running": running,
            "total": self.num_producers,
            "queue_fill": self.get_queue_fill(),
            "restarts": list(self._restart_counts),
        }

    def flush_epoch(self):
        """Epoch barrier: drain queue and advance shared epoch counter.

        Call between training epochs to prevent batch leakage from fast
        producers racing ahead of the consumer.
        """
        self._drain_queue()
        if self._shared_epoch is not None:
            self._shared_epoch.value = self._current_epoch

    def get_queue_fill(self) -> int:
        """Approximate number of batches currently in the queue."""
        if self._data_queue is None:
            return -1
        try:
            return self._data_queue.qsize()
        except (NotImplementedError, Exception):
            return -1

    def __iter__(self):
        """
        Yield up to one epoch of batches.

        If all producers die before the epoch completes, iteration terminates
        early and the epoch counter is NOT incremented (the epoch failed).
        """
        batches_yielded = 0
        batch_count = 0

        running = sum(
            1 for p in self._producer_processes
            if p is not None and p.is_alive()
        )
        if running == 0:
            logging.error(
                "[AsyncSpawnLoader] No producers running; did you call start()?"
            )
            return

        while batches_yielded < self._batches_per_epoch:
            if self._stop_event is not None and self._stop_event.is_set():
                return

            # Periodic health check every 100 batches.
            if self.auto_restart and batch_count > 0 and batch_count % 100 == 0:
                self._restart_failed_producers()
                running = sum(
                    1 for p in self._producer_processes
                    if p is not None and p.is_alive()
                )
                if running == 0:
                    logging.warning(
                        "[AsyncSpawnLoader] All producers died at epoch %d, "
                        "batch %d/%d — epoch truncated",
                        self._current_epoch, batches_yielded, self._batches_per_epoch,
                    )
                    return

            try:
                get_start = time.time()
                item = self._data_queue.get(timeout=2.0)
                get_time = time.time() - get_start

                # --- epoch-tag gate ---
                epoch_tag, sdf_data, indices = item
                if epoch_tag != self._current_epoch:
                    logging.debug(
                        "[AsyncSpawnLoader] Discarded epoch-%d batch (current=%d)",
                        epoch_tag, self._current_epoch,
                    )
                    continue  # don't count toward _batches_per_epoch
                # -----------------------

                batches_yielded += 1
                batch_count += 1

                queue_fill = self.get_queue_fill()
                if batches_yielded % 10 == 0 or get_time > 0.1:
                    print(
                        f"[{time.strftime('%H:%M:%S')}] [AsyncSpawnLoader-Consumer] "
                        f"Got batch {batches_yielded}/{self._batches_per_epoch}, "
                        f"queue_fill={queue_fill}, get_time={get_time:.3f}s"
                    )

                yield sdf_data, indices

            except Empty:
                running = sum(
                    1 for p in self._producer_processes
                    if p is not None and p.is_alive()
                )
                if running == 0:
                    logging.warning(
                        "[AsyncSpawnLoader] All producers died at epoch %d, "
                        "batch %d/%d — epoch truncated",
                        self._current_epoch, batches_yielded, self._batches_per_epoch,
                    )
                    return
                continue

        # Full epoch succeeded — advance counter so restarted producers align.
        self._current_epoch += 1
        if self._shared_epoch is not None:
            self._shared_epoch.value = self._current_epoch

    def __len__(self) -> int:
        return self._batches_per_epoch

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop(immediate=(exc_type is KeyboardInterrupt))
        return False
