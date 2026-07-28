"""
common/cli_logging.py
======================
CLI transcript mirroring (_Tee), per-epoch CSV logging, and training-curve
plotting — extracted verbatim from the identical trio duplicated across
every v2 training script (the specific fieldnames, append-mode + session-
header behavior, and 2-panel accuracy/loss plot layout are all unchanged).
"""
import csv
import sys
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class _Tee:
    """Mirrors stdout to a persistent, append-mode .txt file beside the
    calling script's output folder, with a timestamped session header on
    each open — so restarts/reruns accumulate in one file instead of
    overwriting it."""
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "a", encoding="utf-8")
        self.log.write(f"\n{'='*70}\n[Session start] "
                        f"{datetime.now().isoformat(timespec='seconds')}\n{'='*70}\n")
        self.log.flush()

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()  # flush immediately so a crash/kill doesn't lose the tail

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def plot_history(history: dict, path: str, img_size: int, title: str):
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


def save_log(history: dict, path: str):
    fieldnames = [
        "epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr",
        "epoch_time_s", "vram_peak_alloc_gb", "vram_peak_reserved_gb",
        "cuda_util_pct", "gpu_temp_c", "cpu_pct", "ram_used_gb",
    ]
    n = len(history["train_loss"])
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(n):
            writer.writerow({
                "epoch":            i + 1,
                "train_loss":       f"{history['train_loss'][i]:.6f}",
                "train_acc":        f"{history['train_acc'][i]:.6f}",
                "val_loss":         f"{history['val_loss'][i]:.6f}",
                "val_acc":          f"{history['val_acc'][i]:.6f}",
                "lr":               f"{history['lr'][i]:.8f}",
                "epoch_time_s":     history.get("epoch_time_s",    ["-1"]*n)[i],
                "vram_peak_alloc_gb":    history.get("vram_peak_alloc_gb",   ["-1"]*n)[i],
                "vram_peak_reserved_gb": history.get("vram_peak_reserved_gb",["-1"]*n)[i],
                "cuda_util_pct":    history.get("cuda_util_pct",   ["-1"]*n)[i],
                "gpu_temp_c":       history.get("gpu_temp_c",      ["-1"]*n)[i],
                "cpu_pct":          history.get("cpu_pct",         ["-1"]*n)[i],
                "ram_used_gb":      history.get("ram_used_gb",     ["-1"]*n)[i],
            })
    print(f"[Log] Saved to {path}")
