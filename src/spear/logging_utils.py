
import csv
import logging
import os
import socket
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class _MaxLevelFilter(logging.Filter):
    def __init__(self, max_level: int) -> None:
        super().__init__()
        self._max_level = max_level

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno <= self._max_level


def build_run_context(run_name: str) -> Dict[str, Optional[str]]:
    cwd = os.getcwd()
    host = socket.gethostname()
    user = os.getenv("USER") or os.getenv("LOGNAME") or ""
    slurm_job = os.getenv("SLURM_JOB_ID")
    slurm_task = os.getenv("SLURM_ARRAY_TASK_ID")
    timestamp = datetime.utcnow().isoformat() + "Z"
    return {
        "run_name": run_name,
        "user": user,
        "cwd": cwd,
        "host": host,
        "timestamp_utc": timestamp,
        "slurm_job": slurm_job,
        "slurm_task": slurm_task,
    }


def configure_logging(
    logs_dir: Path, run_name: str, log_level: int = logging.INFO
) -> Tuple[Path, Optional[Dict[str, Optional[str]]]]:
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{run_name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setFormatter(formatter)

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    stdout_handler.addFilter(_MaxLevelFilter(logging.INFO))
    stdout_handler.setFormatter(formatter)

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.WARNING)
    stderr_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(log_level)
    root.addHandler(file_handler)
    root.addHandler(stdout_handler)
    root.addHandler(stderr_handler)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("numexpr").setLevel(logging.WARNING)

    # Emit an initial run-context record so the file log contains node/time info
    run_context: Optional[Dict[str, Optional[str]]] = None
    try:
        run_context = build_run_context(run_name)
        root.info(
            "Run context | run_name=%s | user=%s | cwd=%s | host=%s | timestamp_utc=%s | slurm_job=%s | slurm_task=%s",
            run_context.get("run_name"),
            run_context.get("user"),
            run_context.get("cwd"),
            run_context.get("host"),
            run_context.get("timestamp_utc"),
            run_context.get("slurm_job"),
            run_context.get("slurm_task"),
        )
    except Exception:
        # Never fail logging setup due to environment introspection
        run_context = None

    return log_path, run_context


def get_logger(name: Optional[str] = None) -> logging.Logger:
    return logging.getLogger(name if name else __name__)


try:  # Optional dependency for resource tracking
    import psutil  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    psutil = None  # type: ignore

try:  # Optional GPU visibility
    import torch  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    torch = None  # type: ignore

try:  # Optional GPU utilization via NVML
    import pynvml  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    pynvml = None  # type: ignore


@dataclass
class _ResourceSample:
    time_sec: float
    rss_gib: float
    cpu_percent: float
    thread_count: int
    gpu_memory_gib: float
    gpu_reserved_gib: float
    gpu_util_percent: float


_NVML_INITIALIZED = False
_NVML_DEVICE_HANDLES: List[Any] = []
_NVML_DEVICE_HANDLES_LOCK = threading.Lock()


def _get_gpu_utilization_percent() -> float:
    global _NVML_INITIALIZED
    if pynvml is None or torch is None:
        return float("nan")
    try:
        if not torch.cuda.is_available():
            return float("nan")
        device_count = torch.cuda.device_count()
        if device_count <= 0:
            return float("nan")
        with _NVML_DEVICE_HANDLES_LOCK:
            if not _NVML_INITIALIZED:
                pynvml.nvmlInit()
                _NVML_INITIALIZED = True
                handles: List[Any] = []
                for idx in range(device_count):
                    handles.append(pynvml.nvmlDeviceGetHandleByIndex(idx))
                _NVML_DEVICE_HANDLES.clear()
                _NVML_DEVICE_HANDLES.extend(handles)
            handles_snapshot = list(_NVML_DEVICE_HANDLES)
        utils = []
        for handle in handles_snapshot:
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            utils.append(float(util.gpu))
        if not utils:
            return float("nan")
        return float(sum(utils) / len(utils))
    except Exception:  # pragma: no cover - best-effort only
        return float("nan")


class ResourceUsageTracker:
    """Best-effort background sampler that records resource usage and writes plots/csv."""

    def __init__(self, name: str, output_dir: Path | str, interval_seconds: float = 60.0) -> None:
        self._name = name
        self._safe_name = name.replace(" ", "_")
        self._output_dir = Path(output_dir).expanduser().resolve()
        self._interval = max(1.0, float(interval_seconds))
        self._log = logging.getLogger(__name__)
        self._enabled = psutil is not None
        self._records: List[_ResourceSample] = []
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._start_time: Optional[float] = None
        self._process = psutil.Process() if psutil is not None else None

    def __enter__(self) -> "ResourceUsageTracker":
        if not self._enabled or self._process is None:
            self._log.debug("ResourceUsageTracker disabled (psutil unavailable)")
            return self
        self._start_time = time.time()
        try:
            self._process.cpu_percent(interval=None)
        except Exception:  # pragma: no cover - psutil quirks
            pass
        self._thread = threading.Thread(
            target=self._run,
            name=f"ResourceUsageTracker[{self._safe_name}]",
            daemon=True,
        )
        self._thread.start()
        self._log.debug("ResourceUsageTracker started for %s", self._name)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if not self._enabled or self._process is None:
            return
        self._stop_event.set()
        try:
            self._sample_once()
        except Exception:
            pass
        if self._thread:
            self._thread.join(timeout=self._interval)
        if not self._records:
            self._log.debug("No resource samples captured for %s", self._name)
            return
        self._write_csv()
        self._write_plot()

    def _run(self) -> None:
        if self._process is None:
            return
        while not self._stop_event.is_set():
            try:
                self._sample_once()
            except Exception:  # pragma: no cover - sampling should not crash
                self._log.debug("Resource sample failed for %s", self._name, exc_info=True)
            if self._stop_event.wait(self._interval):
                break

    def _sample_once(self) -> None:
        if self._process is None or self._start_time is None:
            return
        now = time.time()
        rel = now - self._start_time
        try:
            rss_bytes = self._process.memory_info().rss
        except Exception:
            rss_bytes = float("nan")
        try:
            cpu_percent = self._process.cpu_percent(interval=None)
        except Exception:
            cpu_percent = float("nan")
        try:
            thread_count = self._process.num_threads()
        except Exception:
            thread_count = -1

        gpu_memory = 0.0
        gpu_reserved = 0.0
        gpu_util_percent = float("nan")
        if torch is not None:
            try:
                if torch.cuda.is_available():
                    for idx in range(torch.cuda.device_count()):
                        gpu_memory += float(torch.cuda.memory_allocated(idx)) / (1024 ** 3)
                        gpu_reserved += float(torch.cuda.memory_reserved(idx)) / (1024 ** 3)
            except Exception:  # pragma: no cover - defensive GPU handling
                pass
        gpu_util_percent = _get_gpu_utilization_percent()

        self._records.append(
            _ResourceSample(
                time_sec=float(rel),
                rss_gib=float(rss_bytes) / (1024 ** 3) if rss_bytes == rss_bytes else float("nan"),
                cpu_percent=float(cpu_percent),
                thread_count=int(thread_count),
                gpu_memory_gib=float(gpu_memory),
                gpu_reserved_gib=float(gpu_reserved),
                gpu_util_percent=float(gpu_util_percent),
            )
        )

    def _write_csv(self) -> None:
        output_dir = self._output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"{self._safe_name}_resource_usage.csv"
        try:
            with path.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    [
                        "time_sec",
                        "rss_gib",
                        "cpu_percent",
                        "thread_count",
                        "gpu_memory_gib",
                        "gpu_reserved_gib",
                        "gpu_util_percent",
                    ]
                )
                for sample in self._records:
                    writer.writerow(
                        [
                            f"{sample.time_sec:.2f}",
                            f"{sample.rss_gib:.4f}",
                            f"{sample.cpu_percent:.2f}",
                            sample.thread_count,
                            f"{sample.gpu_memory_gib:.4f}",
                            f"{sample.gpu_reserved_gib:.4f}",
                            f"{sample.gpu_util_percent:.2f}",
                        ]
                    )
            self._log.info("Wrote resource usage CSV to %s", path)
        except Exception:  # pragma: no cover - IO errors shouldn't crash pipeline
            self._log.warning("Failed to write resource usage CSV to %s", path, exc_info=True)

    def _write_plot(self) -> None:
        try:
            import matplotlib.pyplot as plt  # type: ignore
        except Exception:  # pragma: no cover - matplotlib optional
            self._log.debug("Skipping resource usage plot for %s (matplotlib unavailable)", self._name)
            return

        times = [sample.time_sec for sample in self._records]
        rss = [sample.rss_gib for sample in self._records]
        cpu = [sample.cpu_percent for sample in self._records]
        gpu = [sample.gpu_memory_gib for sample in self._records]
        gpu_reserved = [sample.gpu_reserved_gib for sample in self._records]
        gpu_util = [sample.gpu_util_percent for sample in self._records]
        threads = [sample.thread_count for sample in self._records]

        if not times:
            return

        output_dir = self._output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        time_min = min(times)
        time_max = max(times)

        series_to_plot: list[tuple[str, list[float], str, str]] = [
            ("RSS (GiB)", rss, "RSS (GiB)", "#1f77b4"),
            ("CPU %", cpu, "CPU %", "#ff7f0e"),
        ]
        if any(value > 0.0 for value in gpu):
            series_to_plot.append(("GPU mem (GiB)", gpu, "GPU mem (GiB)", "#2ca02c"))
        if any(value > 0.0 for value in gpu_reserved):
            series_to_plot.append(("GPU reserved (GiB)", gpu_reserved, "GPU reserved (GiB)", "#17becf"))
        if any(value == value for value in gpu_util):
            series_to_plot.append(("GPU %", gpu_util, "GPU %", "#9467bd"))
        if any(value > 0 for value in threads):
            series_to_plot.append(("Threads", threads, "Threads", "#8c564b"))

        if not series_to_plot:
            return

        cols = 2
        rows = int(np.ceil(len(series_to_plot) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 3.8 * rows), sharex=True)
        axes = np.atleast_1d(axes).ravel()

        for ax, (label, values, ylabel, color) in zip(axes, series_to_plot):
            ax.plot(times, values, color=color, linewidth=2.0)
            ax.set_title(label)
            ax.set_ylabel(ylabel)
            ax.set_xlim(time_min, time_max)
            ax.grid(True, alpha=0.3)
        for ax in axes[len(series_to_plot):]:
            ax.axis("off")

        for ax in axes:
            ax.set_xlabel("Time (s)")

        fig.suptitle(f"Resource usage | {self._name}", y=0.98)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        path = output_dir / f"{self._safe_name}_resource_usage.png"
        try:
            fig.savefig(path, dpi=300)
            self._log.info("Wrote resource usage plot to %s", path)
        except Exception:  # pragma: no cover - IO errors shouldn't crash pipeline
            self._log.warning("Failed to write resource usage plot to %s", path, exc_info=True)
        finally:
            plt.close(fig)
