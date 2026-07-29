"""
common/cli_logging.py
======================
CLI transcript mirroring (_Tee), per-epoch CSV logging, and training-curve
plotting — extracted verbatim from the identical trio duplicated across
every v2 training script (the specific fieldnames, append-mode + session-
header behavior, and 2-panel accuracy/loss plot layout are all unchanged).

DDP note (2026-07-28, per direct user follow-up — see common/
distributed.py, including its "NOT YET VALIDATED against real multi-GPU
hardware" caveat, which applies here too): _Tee/save_log()/plot_history()
are all rank-0-only now. Every rank's own console stdout still prints
normally (useful for spotting a hung/crashed non-rank-0 process) — only
the FILE writes (transcript .txt, CSV, .png) are suppressed on every
rank but 0, since several ranks all appending to or overwriting the same
path would race and could corrupt the file. On a single-process run
(the default — no torchrun involved) this is a complete no-op: rank 0 of
world size 1 is exactly what every existing invocation already was.
"""
import csv
import sys
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .distributed import is_main_process


class _Tee:
    """Mirrors stdout to a persistent, append-mode .txt file beside the
    calling script's output folder, with a timestamped session header on
    each open — so restarts/reruns accumulate in one file instead of
    overwriting it. Rank-0-only file writes on a distributed run (see
    module docstring) — every rank's console output still passes through
    unchanged; self.log is None on non-rank-0, and write()/flush() guard
    on that rather than assuming it's always an open file handle."""
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = None
        if is_main_process():
            self.log = open(filepath, "a", encoding="utf-8")
            self.log.write(f"\n{'='*70}\n[Session start] "
                            f"{datetime.now().isoformat(timespec='seconds')}\n{'='*70}\n")
            self.log.flush()

    def write(self, message):
        self.terminal.write(message)
        if self.log is not None:
            self.log.write(message)
            self.log.flush()  # flush immediately so a crash/kill doesn't lose the tail

    def flush(self):
        self.terminal.flush()
        if self.log is not None:
            self.log.flush()


def plot_history(history: dict, path: str, img_size: int, title: str):
    if not is_main_process():
        return
    ep = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"{title} {img_size}x{img_size} — Training History",
                 fontsize=12, fontweight="bold")
    ax1.plot(ep, history["train_acc"], "b-o", markersize=4, label="Train")
    ax1.plot(ep, history["val_acc"],   "r-o", markersize=4, label="Val")
    ax1.set_title("Accuracy"); ax1.set_xlabel("Epoch"); ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2.plot(ep, history["train_loss"], "b-o", markersize=4, label="Train")
    ax2.plot(ep, history["val_loss"],   "r-o", markersize=4, label="Val")
    ax2.set_title("Loss"); ax2.set_xlabel("Epoch"); ax2.legend(); ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150); plt.close()
    print(f"[Plot] Saved to {path}")


# Per-epoch hardware telemetry columns (2026-07-28, per direct user
# follow-up — expanded from the original 4-field cuda_util_pct/gpu_temp_c/
# cpu_pct/ram_used_gb set). Sourced from HardwareMonitor.epoch_summary()
# (common/telemetry.py) plus the two data_wait_s/compute_s fields each
# training script's own train_one_epoch() times directly (see that
# module's docstring for why that timing lives in the caller, not here).
# Listed as one constant so save_log() and any caller building a history
# dict agree on the exact same column set without repeating it.
HW_TELEMETRY_FIELDS = [
    "cuda_util_pct_min", "cuda_util_pct_avg", "cuda_util_pct_max",
    "cuda_mem_util_pct_min", "cuda_mem_util_pct_avg", "cuda_mem_util_pct_max",
    "gpu_temp_c_min", "gpu_temp_c_avg", "gpu_temp_c_max",
    "gpu_power_w_min", "gpu_power_w_avg", "gpu_power_w_max",
    "gpu_clock_sm_mhz_avg", "gpu_clock_mem_mhz_avg",
    "gpu_fan_pct_avg", "gpu_throttled_any",
    "cpu_pct_min", "cpu_pct_avg", "cpu_pct_max",
    "ram_used_gb_min", "ram_used_gb_avg", "ram_used_gb_max", "ram_total_gb",
    "disk_read_mb_s_min", "disk_read_mb_s_avg", "disk_read_mb_s_max",
    "disk_write_mb_s_min", "disk_write_mb_s_avg", "disk_write_mb_s_max",
    "vram_peak_alloc_gb", "vram_peak_reserved_gb",
    "data_wait_s", "compute_s",
]


def save_log(history: dict, path: str):
    if not is_main_process():
        return
    fieldnames = [
        "epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr",
        "epoch_time_s",
    ] + HW_TELEMETRY_FIELDS
    n = len(history["train_loss"])
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(n):
            row = {
                "epoch":            i + 1,
                "train_loss":       f"{history['train_loss'][i]:.6f}",
                "train_acc":        f"{history['train_acc'][i]:.6f}",
                "val_loss":         f"{history['val_loss'][i]:.6f}",
                "val_acc":          f"{history['val_acc'][i]:.6f}",
                "lr":               f"{history['lr'][i]:.8f}",
                "epoch_time_s":     history.get("epoch_time_s", ["-1"]*n)[i],
            }
            for field in HW_TELEMETRY_FIELDS:
                row[field] = history.get(field, ["-1"]*n)[i]
            writer.writerow(row)
    print(f"[Log] Saved to {path}")
