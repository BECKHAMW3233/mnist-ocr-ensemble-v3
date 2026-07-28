"""
common/checkpointing.py
========================
Early stopping (with best-checkpoint saving) and the resume-state save
helper, extracted verbatim from the identical EarlyStopping class and
try/except-wrapped torch.save() resume block duplicated across every v2
training script.

Resume/checkpoint correctness (preserved from v2, see v3_CHANGELOG.md for
the full precedent): a resume file records optimizer/scheduler/scaler
state, patience counter, best val loss, epoch, and history. Each caller
is still responsible for validating that a loaded resume file's own
recorded batch size (where applicable — see the router's LR-scaling
resume check, mirroring v2's digit-gate) matches the batch size THIS run
auto-detected before trusting optimizer/scheduler state — auto-detected
batch size is not guaranteed stable run-to-run (different machine,
different background VRAM/RAM load), and blindly resuming
optimizer/scheduler state built for a different batch size can silently
carry forward a wrong-for-this-run LR schedule. That check is caller-
specific (which state pieces are batch-size-dependent differs by
optimizer) and is therefore NOT generalized here — see each training
script's own resume block.

DDP note (2026-07-28, per direct user follow-up — see common/
distributed.py for the full multi-GPU picture, including the "NOT YET
VALIDATED against real multi-GPU hardware" caveat that applies here too):
every file write in this module is now gated to rank 0 only (a no-op,
not an error, on every other rank) — with several ranks all training
against the same checkpoint/resume path, letting every rank write it
independently would race, and the loser's write could corrupt or
truncate the file mid-read by another process. EarlyStopping's actual
DECISION logic (best_loss/counter/stop) deliberately still runs
identically on EVERY rank, not just rank 0 — every rank's `if
early_stop.stop: break` must agree, or a rank that breaks while another
doesn't will hang the others waiting on a collective call (e.g. the next
epoch's metric all-reduce) that never comes. This works correctly
without any extra synchronization ONLY because the val_loss fed into
EarlyStopping.__call__() must already be the all-reduced (globally
correct, identical-on-every-rank) value BEFORE this is called — see
common/distributed.py's all_reduce_sum() and each training script's
evaluate() call site.
"""
from pathlib import Path

import torch
import torch.nn as nn

from .distributed import is_main_process, unwrap_model


class EarlyStopping:
    """
    Tracks best validation loss; saves a checkpoint (state_dict + the
    val_loss that earned it) every time a new best is found. `stop`
    becomes True once `patience` consecutive epochs pass without
    improvement — the caller's training loop is responsible for actually
    breaking out of the epoch loop. val_loss must already be globally
    correct (all-reduced) before being passed in on a distributed run —
    see module docstring.
    """
    def __init__(self, patience: int, path: str):
        self.patience  = patience
        self.path      = path
        self.best_loss = float("inf")
        self.counter   = 0
        self.stop      = False

    def __call__(self, val_loss: float, model: nn.Module):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter   = 0
            if is_main_process():
                torch.save({"state_dict": unwrap_model(model).state_dict(),
                            "val_loss": val_loss}, self.path)
                print(f"  [Checkpoint] val_loss → {val_loss:.4f}  saved")
        else:
            self.counter += 1
            if is_main_process():
                print(f"  [EarlyStopping] {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.stop = True
                if is_main_process():
                    print("  [EarlyStopping] Halting.")


def save_resume_state(path, **state) -> None:
    """
    Wraps torch.save() for the per-epoch resume file in the same
    try/except every v2 script used — a failed resume write (e.g. a
    transient disk issue) prints a warning and lets training continue
    rather than crashing an otherwise-healthy run. Rank-0-only (see
    module docstring) — a no-op on every other rank.
    """
    if not is_main_process():
        return
    try:
        torch.save(state, str(path))
    except Exception as e:
        print(f"  [Resume] Warning: could not save state: {e}")


def clear_resume_state(path) -> None:
    """Deletes the resume file once training completes successfully —
    matches every v2 script's own cleanup-on-success behavior. Rank-0-
    only (see module docstring) — a no-op on every other rank."""
    if not is_main_process():
        return
    resume_p = Path(path)
    if resume_p.exists():
        resume_p.unlink()
        print("  [Resume] State cleared — training complete.")
