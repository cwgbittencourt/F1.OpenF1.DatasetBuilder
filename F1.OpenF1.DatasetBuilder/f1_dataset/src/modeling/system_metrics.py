from __future__ import annotations

import os
import time
from dataclasses import dataclass

try:  # pragma: no cover - defensive import for environments without psutil
    import psutil
except Exception:  # pragma: no cover
    psutil = None


@dataclass
class SystemMetrics:
    start_time: float
    start_cpu_time: float
    cpu_count: int
    process: object | None

    @classmethod
    def start(cls) -> "SystemMetrics":
        cpu_count = os.cpu_count() or 1
        start_time = time.perf_counter()
        if psutil is None:
            return cls(start_time, 0.0, cpu_count, None)
        proc = psutil.Process()
        try:
            proc.cpu_percent(interval=None)
        except Exception:
            pass
        try:
            cpu_times = proc.cpu_times()
            start_cpu = float(cpu_times.user + cpu_times.system)
        except Exception:
            start_cpu = 0.0
        return cls(start_time, start_cpu, cpu_count, proc)

    def collect(self) -> dict[str, float]:
        elapsed = max(time.perf_counter() - self.start_time, 0.0)
        if not self.process or psutil is None:
            return {"sys_duration_sec": float(elapsed)}

        metrics: dict[str, float] = {"sys_duration_sec": float(elapsed)}
        try:
            cpu_times = self.process.cpu_times()
            cpu_time = float(cpu_times.user + cpu_times.system)
            cpu_time_delta = max(cpu_time - self.start_cpu_time, 0.0)
            metrics["sys_cpu_time_sec"] = cpu_time_delta
            if elapsed > 0 and self.cpu_count > 0:
                metrics["sys_cpu_percent"] = (cpu_time_delta / elapsed) * 100.0 / self.cpu_count
        except Exception:
            pass

        try:
            mem = self.process.memory_info()
            metrics["sys_mem_rss_mb"] = float(mem.rss) / (1024 * 1024)
            metrics["sys_mem_vms_mb"] = float(mem.vms) / (1024 * 1024)
        except Exception:
            pass

        try:
            metrics["sys_num_threads"] = float(self.process.num_threads())
        except Exception:
            pass

        try:
            io = self.process.io_counters()
            metrics["sys_io_read_mb"] = float(io.read_bytes) / (1024 * 1024)
            metrics["sys_io_write_mb"] = float(io.write_bytes) / (1024 * 1024)
        except Exception:
            pass

        return metrics
