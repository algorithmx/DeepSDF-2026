#!/usr/bin/env python3
"""
System watchdog for training monitoring.

Samples CPU/GPU metrics in a background daemon thread and writes
timestamped tabular logs to a file and (optionally) to stderr.

Usage:
    from deep_sdf.watchdog import SystemMonitor

    monitor = SystemMonitor(interval=2.0, log_path="/tmp/watchdog.log")
    monitor.start()
    # ... training ...
    monitor.stop()
"""

import os
import time
import logging
import threading
from datetime import datetime
from typing import List, Optional

logger = logging.getLogger(__name__)


class SystemMonitor:
    """Background thread that samples CPU/GPU at a fixed interval."""

    def __init__(self, interval: float = 2.0, log_path: Optional[str] = None, gpu_ids: Optional[List[int]] = None):
        """
        Args:
            interval: Sampling period in seconds.
            log_path: Path to the log file.  If None, logs to stderr only.
            gpu_ids: List of GPU indices to monitor.  If None, monitors all.
        """
        self.interval = interval
        self.log_path = log_path
        self.gpu_ids = gpu_ids
        self._stop_event = threading.Event()
        self._thread = None
        self._file = None
        self._nvml_handle = None
        self._gpu_count = 0

    def _init_nvml(self):
        try:
            import pynvml
            pynvml.nvmlInit()
            self._gpu_count = (
                len(self.gpu_ids) if self.gpu_ids else pynvml.nvmlDeviceGetCount()
            )
            self._nvml_handle = pynvml
        except Exception as e:
            logger.warning(f"NVML init failed: {e}")
            self._nvml_handle = None

    def _shutdown_nvml(self):
        if self._nvml_handle is not None:
            try:
                self._nvml_handle.nvmlShutdown()
            except Exception:
                pass

    def _sample_cpu(self):
        import psutil
        per_cpu = psutil.cpu_percent(interval=0, percpu=True)
        overall = psutil.cpu_percent(interval=0)
        mem = psutil.virtual_memory()
        return overall, per_cpu, mem

    def _sample_gpu(self):
        if self._nvml_handle is None:
            return []
        nvml = self._nvml_handle
        gpu_ids = self.gpu_ids or range(self._gpu_count)
        samples = []
        for gid in gpu_ids:
            try:
                handle = nvml.nvmlDeviceGetHandleByIndex(gid)
                util = nvml.nvmlDeviceGetUtilizationRates(handle)
                mem_info = nvml.nvmlDeviceGetMemoryInfo(handle)
                try:
                    temp = nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU)
                except Exception:
                    temp = -1
                samples.append({
                    "id": gid,
                    "gpu_util": util.gpu,
                    "mem_util": util.memory,
                    "vram_used": mem_info.used / (1024 ** 3),
                    "vram_total": mem_info.total / (1024 ** 3),
                    "temp": temp,
                })
            except Exception as e:
                logger.debug(f"GPU {gid} sample failed: {e}")
                samples.append({"id": gid, "gpu_util": -1, "mem_util": -1,
                                "vram_used": 0, "vram_total": 0, "temp": -1})
        return samples

    def _format_line(self, ts, cpu_overall, cpu_per, mem, gpu_samples):
        parts = [ts]
        parts.append(f"CPU {cpu_overall:5.1f}%")
        parts.append(f"MEM {mem.used / (1024**3):5.1f}/{mem.total / (1024**3):.1f}GB ({mem.percent:4.1f}%)")
        for g in gpu_samples:
            tag = f"GPU{g['id']}"
            if g["gpu_util"] < 0:
                parts.append(f"{tag} ERR")
            else:
                parts.append(f"{tag} {g['gpu_util']:3d}%")
                parts.append(f"VRAM {g['vram_used']:5.1f}/{g['vram_total']:.1f}GB")
                if g["temp"] >= 0:
                    parts.append(f"{g['temp']}°C")
        return " | ".join(parts)

    def _header(self):
        gpu_ids = self.gpu_ids or list(range(self._gpu_count))
        gpu_cols = " ".join(f"{'GPU'+str(g):>14s} {'VRAM':>18s} {'TEMP':>5s}" for g in gpu_ids)
        return f"{'TIMESTAMP':>19s} | {'CPU':>7s} | {'MEMORY':>20s} | {gpu_cols}"

    def _run(self):
        import psutil

        self._init_nvml()

        if self.log_path:
            os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)
            self._file = open(self.log_path, "w")
            header = self._header()
            self._file.write(header + "\n")
            self._file.flush()

        # First call to initialize psutil's baseline
        psutil.cpu_percent(interval=0, percpu=True)

        while not self._stop_event.is_set():
            try:
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                cpu_overall, cpu_per, mem = self._sample_cpu()
                gpu_samples = self._sample_gpu()
                line = self._format_line(ts, cpu_overall, cpu_per, mem, gpu_samples)

                if self._file:
                    self._file.write(line + "\n")
                    self._file.flush()
                logger.debug(line)

            except Exception as e:
                logger.debug(f"Watchdog sample error: {e}")

            self._stop_event.wait(self.interval)

        if self._file:
            self._file.close()

        self._shutdown_nvml()

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="watchdog")
        self._thread.start()
        logger.info(f"Watchdog started (interval={self.interval}s, log={self.log_path})")

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval * 2)
            self._thread = None
        logger.info("Watchdog stopped")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()
