"""
v3_mnist_digit_soap_64.py
====================
MNIST digit ensemble — OCRConvNetTriplePyramid, SOAP optimizer, 64x64
Pure PyTorch — Triple-width channels + Multi-Scale feature pyramid fusion.
Adapted from v2's mnist_soap_64.py (32/64/128 tiers) — the
16x16 tier is new for v3, no v2 counterpart. See v3_CHANGELOG.md for the
full v3 restructure (optimizer roster change, modularization, naming).

Architecture — OCRConvNetTriplePyramid (unchanged from v2):
  Channel progression: 96→192→384→768
  Feature pyramid: concatenates pooled outputs from stages 2+3+4 (fused_dim=1344)
  Classifier head: 1344→1024→512→256→128→10 (5 layers, GELU)
  SE reduction: 16, drop_path=0.05, dropout=0.35 (first classifier layer)
  label_smoothing=0.05
  ~4.6M parameters

Install: pip install pytorch_optimizer psutil

OPTIMIZER — SOAP (Shampoo + Adam, Kronecker-factored second-order). Kept
in the v3 roster: still the strongest real-world accuracy per prior
validation, at a real but bounded per-step compute cost — see
v3_CHANGELOG.md's roster-change entry for why SOAP was kept while Lion,
SGD, and AdaHessian were dropped.
  lr=1e-3, betas=(0.95, 0.95), weight_decay=5e-4
  precondition_frequency=100 — Kronecker factor update every 100 steps
    (unchanged from v2 — see v3_CHANGELOG.md for the original tuning notes).
  500-step linear warmup before cosine decay (common/scheduler.py)
  PATIENCE=20
  Standard first-order backward — no create_graph needed.
  AMP disabled — Kronecker eigendecomposition requires float32.

Digit sources at this resolution tier: see
supplementary_data.digit_sources_for_tier() — USPS is the ENTIRE
training spine at 16x16 (via load_base_usps(), not load_base_mnist() —
see load_mnist() below), since it's USPS's own native resolution and the
only source in that tier's ladder; MNIST/EMNIST Digits/SVHN/ARDIS IV
(every source except USPS) at 28x28; every source including USPS at
32/64/128. See Part 1 of the v3 restructure / v3_CHANGELOG.md for why there
are 5 resolution-tagged files per optimizer (16/28/32/64/128), not 4.

Augmentation: rotation ±5°, affine, color jitter, light blur — unchanged
  from v2, see get_transforms() below.

Output: ./v3_mnist_digit_soap_64/  (created next to this script)
"""

# =============================================================================
# 0. IMPORTS
# =============================================================================
import sys
import argparse
import time
from pathlib import Path
from datetime import datetime

from common.seeding import (
    apply_cublas_workspace_config, get_global_seed, set_all_seeds, reserve_cpu_threads,
)

apply_cublas_workspace_config()

import torch

reserve_cpu_threads()
GLOBAL_SEED = get_global_seed()
set_all_seeds(GLOBAL_SEED)

import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler, ConcatDataset


from common.telemetry import get_hw_stats, setup_device, HAS_PSUTIL
from common.batch_sizing import determine_batch_size
from common.checkpointing import EarlyStopping, save_resume_state, clear_resume_state
from common.cli_logging import _Tee, plot_history, save_log
from common.onnx_export import export_onnx
from common.scheduler import WarmupCosineScheduler

if HAS_PSUTIL:
    import psutil

try:
    from pytorch_optimizer import SOAP
except ImportError:
    print("ERROR: pytorch_optimizer not installed. Run: pip install pytorch_optimizer")
    sys.exit(1)

try:
    from supplementary_data import (
        load_supplementary, get_combined_weights, load_base_mnist, load_base_usps,
        digit_sources_for_tier,
    )
    HAS_SUPPLEMENTARY = True
except ImportError:
    HAS_SUPPLEMENTARY = False
    print("[Warning] supplementary_data.py not found — digit supplementary data unavailable")

import torchvision.transforms as transforms


# =============================================================================
# 1. CONFIGURATION
# =============================================================================

RAM_RESERVE_GB   = 4.0

DATA_DIR      = Path(r"E:\CSC-114\emnist-model\datasets\pytorch")
# ^ Specific to the original project machine — change this to wherever
# YOU want MNIST/EMNIST/USPS/SVHN to be downloaded/read from on your own
# system (see supplementary_data.py's own DATA_DIR for the canonical
# version of this path; this local copy must be kept in sync with it
# manually, matching v2's own convention of each script carrying its own
# DATA_DIR literal).

NUM_CLASSES   = 10
LABEL_MAP     = list("0123456789")
PATIENCE      = 20
LR            = 1e-3
BETAS         = (0.95, 0.95)
WEIGHT_DECAY  = 5e-4
PRECOND_FREQ  = 100
WARMUP_STEPS  = 500
LABEL_SMOOTH  = 0.05
DROP_PATH     = 0.05
DROPOUT_HEAD  = 0.35
SCHEDULE_EPOCH_ESTIMATE = 200  # shapes the cosine LR curve only, not a hard epoch cap —
                                # PATIENCE-based early stopping is what actually ends training.

IMG_SIZE      = 64
# OUTPUT_ROOT is script-relative (Path(__file__).resolve().parent / ...) —
# safe as-is on any machine, unlike DATA_DIR above. It creates its own
# output folder next to wherever this script actually runs from, so it
# does not need to be personalized for a different system.
OUTPUT_ROOT   = Path(__file__).resolve().parent / f"v3_mnist_digit_soap_{IMG_SIZE}"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_config(img_size: int, batch_size: int) -> dict:
    prefix = f"v3_mnist_digit_soap_{img_size}"
    return {
        "img_size":    img_size,
        "batch_size":  batch_size,
        "checkpoint_path":  str(OUTPUT_ROOT / f"{prefix}_best.pt"),
        "final_model_path": str(OUTPUT_ROOT / f"{prefix}_final.pt"),
        "onnx_path":        str(OUTPUT_ROOT / f"{prefix}.onnx"),
        "log_path":         str(OUTPUT_ROOT / f"{prefix}_log.csv"),
        "plot_path":        str(OUTPUT_ROOT / f"{prefix}_curves.png"),
        "resume_path":      str(OUTPUT_ROOT / f"{prefix}_resume.pt"),
    }


# =============================================================================
# 2. DATA PIPELINE
# =============================================================================

def get_transforms(img_size: int, augment: bool = False) -> transforms.Compose:
    aug_transforms = [
        transforms.RandomRotation(degrees=5),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.85, 1.15), shear=5),
        transforms.ColorJitter(contrast=0.3, brightness=0.1),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.3),
    ] if augment else []
    base_transforms = [transforms.Resize((img_size, img_size)), transforms.ToTensor()]
    return transforms.Compose(aug_transforms + base_transforms)


def load_mnist(data_dir: Path, img_size: int):
    """
    Loads this resolution tier's digit spine plus supplementary sources
    (supplementary_data.digit_sources_for_tier()). At 16x16, USPS IS the
    spine (load_base_usps() — its own native resolution, the only source
    in that tier's ladder); at every other tier, base MNIST is the spine
    (load_base_mnist()) plus whichever supplementary sources
    digit_sources_for_tier() returns for that tier (every source except
    USPS at 28x28; every source at 32/64/128).
    """
    train_transform = get_transforms(img_size, augment=True)
    test_transform  = get_transforms(img_size, augment=False)

    base_loader = load_base_usps if img_size == 16 else load_base_mnist
    train_ds, val_ds, test_ds, train_targets = base_loader(
        data_dir=data_dir,
        train_transform=train_transform,
        val_transform=test_transform,
        test_transform=test_transform,
        validation_split=0.15,
        split_seed=GLOBAL_SEED,
        return_train_targets=True,
    )

    if HAS_SUPPLEMENTARY:
        supp_ds = load_supplementary(
            transform=train_transform,
            train=True,
            **digit_sources_for_tier(img_size),
        )
    else:
        supp_ds = None

    return train_ds, val_ds, test_ds, train_targets, supp_ds


def make_dataloader(train_ds, train_targets, supp_ds, batch_size: int):
    if supp_ds is not None:
        combined       = ConcatDataset([train_ds, supp_ds])
        sample_weights = get_combined_weights(train_ds, supp_ds)
        print(f"[Dataset] Combined: {len(combined):,} samples")
    else:
        combined = train_ds
        class_counts   = torch.bincount(train_targets, minlength=NUM_CLASSES).float().clamp(min=1)
        class_weights  = 1.0 / class_counts
        sample_weights = class_weights[train_targets]

    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    return DataLoader(combined, batch_size=batch_size, sampler=sampler,
                      num_workers=10, pin_memory=True, persistent_workers=True)


def make_eval_dataloader(dataset, batch_size: int):
    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      num_workers=0, pin_memory=True, persistent_workers=False)


# =============================================================================
# 3. ARCHITECTURE — OCRConvNetTriplePyramid (unchanged from v2)
# =============================================================================

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep  = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask  = torch.rand(shape, device=x.device) < keep
        return x * mask.float() / keep


class TripleBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, drop_path=0.0):
        super().__init__()
        mid = out_ch // 2
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, mid, 1, bias=False),
            nn.BatchNorm2d(mid),
            nn.GELU(),
            nn.Conv2d(mid, mid, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.GELU(),
            nn.Conv2d(mid, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.se       = SEBlock(out_ch, reduction=16)
        self.drop     = DropPath(drop_path)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.drop(self.se(self.conv(x))) + self.shortcut(x))


class OCRConvNetTriplePyramid(nn.Module):
    """
    Triple-width channels + multi-scale feature pyramid fusion.
    Input:  (batch, 1, IMG_SIZE, IMG_SIZE)
    Output: (batch, 10)
    """
    def __init__(self, num_classes=10, drop_path=0.05, dropout=0.35):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 96, 3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.GELU(),
        )
        self.stage1 = self._make_stage(96,  96,  2, drop_path)
        self.stage2 = self._make_stage(96,  192, 2, drop_path)
        self.stage3 = self._make_stage(192, 384, 2, drop_path)
        self.stage4 = self._make_stage(384, 768, 2, drop_path)

        self.pool2 = nn.AdaptiveAvgPool2d(1)
        self.pool3 = nn.AdaptiveAvgPool2d(1)
        self.pool4 = nn.AdaptiveAvgPool2d(1)
        fused_dim  = 192 + 384 + 768  # 1344

        self.head = nn.Sequential(
            nn.Linear(fused_dim, 1024),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, num_classes),
        )

    def _make_stage(self, in_ch, out_ch, num_blocks, drop_path):
        layers = [TripleBlock(in_ch, out_ch, stride=2, drop_path=drop_path)]
        for _ in range(num_blocks - 1):
            layers.append(TripleBlock(out_ch, out_ch, stride=1, drop_path=drop_path))
        return nn.Sequential(*layers)

    def forward(self, x):
        x  = self.stem(x)
        x  = self.stage1(x)
        s2 = self.stage2(x)
        s3 = self.stage3(s2)
        s4 = self.stage4(s3)
        f2 = self.pool2(s2).flatten(1)
        f3 = self.pool3(s3).flatten(1)
        f4 = self.pool4(s4).flatten(1)
        return self.head(torch.cat([f2, f3, f4], dim=1))


def build_soap(params):
    return SOAP(params, lr=LR, betas=BETAS, weight_decay=WEIGHT_DECAY,
                precondition_frequency=PRECOND_FREQ)


# =============================================================================
# 4. TRAINING / EVALUATION
# =============================================================================

def train_one_epoch(model, loader, criterion, optimizer, scheduler, device,
                    epoch: int = None, img_size: int = None) -> tuple:
    """No AMP/GradScaler and no gradient clipping — SOAP's Kronecker
    eigendecomposition requires float32, and clipping was never part of
    this optimizer's configuration here. Standard first-order backward."""
    model.train()
    total_loss = total_correct = total_samples = 0
    num_batches  = len(loader)
    midpoint_idx = num_batches // 2
    mid_epoch_hw = None
    log_interval = max(1, round(num_batches * 0.025))

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad()

        logits = model(images)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if batch_idx == midpoint_idx:
            mid_epoch_hw = get_hw_stats()

        total_loss    += loss.item() * images.size(0)
        total_correct += (logits.argmax(1) == labels).sum().item()
        total_samples += images.size(0)

        if (batch_idx + 1) % log_interval == 0 or (batch_idx + 1) == num_batches:
            _running_loss = total_loss / total_samples
            _running_acc  = total_correct / total_samples
            _prefix = f"[{img_size}x{img_size}] " if img_size is not None else ""
            _epoch_str = f"Epoch {epoch:3d}  " if epoch is not None else ""
            print(f"  {_prefix}{_epoch_str}Batch {batch_idx + 1:5d}/{num_batches:5d}  "
                  f"loss: {_running_loss:.4f}  acc: {_running_acc:.4f}")

    if mid_epoch_hw is None:
        mid_epoch_hw = get_hw_stats()
    return total_loss / total_samples, total_correct / total_samples, mid_epoch_hw


@torch.no_grad()
def evaluate(model, loader, criterion, device, per_class: bool = False) -> tuple:
    model.eval()
    total_loss = total_correct = total_samples = 0

    if per_class:
        class_correct = torch.zeros(NUM_CLASSES)
        class_total   = torch.zeros(NUM_CLASSES)

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss   = criterion(logits, labels)
        preds  = logits.argmax(1)

        total_loss    += loss.item() * images.size(0)
        total_correct += (preds == labels).sum().item()
        total_samples += images.size(0)

        if per_class:
            for c in range(NUM_CLASSES):
                mask = labels == c
                class_correct[c] += (preds[mask] == labels[mask]).sum().item()
                class_total[c]   += mask.sum().item()

    if per_class and class_total.sum() > 0:
        class_acc = class_correct / class_total.clamp(min=1)
        worst = class_acc.argsort()[:10]
        print("\n  [Per-Class] 10 worst-performing classes:")
        for idx in worst:
            print(f"    '{LABEL_MAP[idx]}' (class {idx:2d}): "
                  f"{class_acc[idx]*100:.1f}%  ({int(class_total[idx])} samples)")

    return total_loss / total_samples, total_correct / total_samples


# =============================================================================
# 5. TRAINING PROCEDURE
# =============================================================================

def run_training(img_size: int, batch_override: int = None):
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    device = setup_device(use_amp=False)

    if batch_override is not None:
        batch_size = batch_override
        print(f"[Batch] Override: using batch size {batch_size} (skipping auto-detect)")
    else:
        probe_steps = max(5, PRECOND_FREQ // 10)  # SOAP-specific: scales with
        # PRECOND_FREQ to guarantee at least one real Kronecker
        # preconditioning update fires during each probe.
        batch_size = determine_batch_size(
            build_model=lambda: OCRConvNetTriplePyramid(NUM_CLASSES, DROP_PATH, DROPOUT_HEAD),
            build_optimizer=build_soap,
            criterion=nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH),
            img_size=img_size, num_classes=NUM_CLASSES, device=device,
            probe_steps=probe_steps, use_amp=False, grad_clip_norm=None,
        )
    cfg = build_config(img_size, batch_size)

    print("=" * 60)
    print(f"  MNIST Digit Ensemble — SOAP  [{img_size}x{img_size}]")
    print(f"  PyTorch {torch.__version__}  |  AMP: False (SOAP requires float32)")
    print(f"  Output: {OUTPUT_ROOT}")
    print(f"  Resolution: {img_size}x{img_size}  |  Batch: {cfg['batch_size']} {'(override)' if batch_override else '(auto-detected)'}")
    print(f"  Optimizer: SOAP (Shampoo + Adam, Kronecker-factored second-order)")
    print(f"  Digit sources this tier: {digit_sources_for_tier(img_size)}")
    print(f"  Normalization: [0,1]")
    print("=" * 60)

    train_ds, val_ds, test_ds, train_targets, supp_ds = load_mnist(DATA_DIR, img_size)
    train_loader = make_dataloader(train_ds, train_targets, supp_ds, cfg["batch_size"])
    val_loader   = make_eval_dataloader(val_ds,  cfg["batch_size"])
    test_loader  = make_eval_dataloader(test_ds, cfg["batch_size"])

    model = OCRConvNetTriplePyramid(num_classes=NUM_CLASSES, drop_path=DROP_PATH, dropout=DROPOUT_HEAD).to(device)
    total = sum(p.numel() for p in model.parameters())
    print(f"\n[Model] OCRConvNetTriplePyramid — {img_size}x{img_size}")
    print(f"  Parameters : {total:,}")
    print(f"  Est. size  : {total * 4 / 1024**2:.1f} MB (float32)")

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)

    total_steps = SCHEDULE_EPOCH_ESTIMATE * len(train_loader)
    optimizer = build_soap(model.parameters())
    scheduler = WarmupCosineScheduler(optimizer, WARMUP_STEPS, total_steps)
    print(f"[Optimizer] SOAP  lr={LR}  betas={BETAS}  wd={WEIGHT_DECAY}  "
          f"precond_freq={PRECOND_FREQ}")
    print(f"            warmup_steps={WARMUP_STEPS}  total_steps={total_steps:,}")

    early_stop = EarlyStopping(patience=PATIENCE, path=cfg["checkpoint_path"])

    print(f"\n[Train] Starting {img_size}x{img_size} — no epoch cap | batch: {cfg['batch_size']} | patience: {PATIENCE}")
    history = {k: [] for k in ["train_loss", "train_acc", "val_loss", "val_acc", "lr"]}

    # ── Resume from previous session ─────────────────────────────────────────
    start_epoch = 1
    resume_path = Path(cfg["resume_path"])
    if resume_path.exists() and Path(cfg["checkpoint_path"]).exists():
        try:
            _rs = torch.load(str(resume_path), map_location=device, weights_only=False)
            _ckpt = torch.load(cfg["checkpoint_path"], map_location=device, weights_only=False)
            model.load_state_dict(_ckpt["state_dict"] if "state_dict" in _ckpt else _ckpt)
            optimizer.load_state_dict(_rs["optimizer_state"])
            scheduler.load_state_dict(_rs["scheduler_state"])
            early_stop.counter   = _rs["patience_counter"]
            early_stop.best_loss = _rs["best_val_loss"]
            history               = _rs["history"]
            start_epoch           = _rs["epoch"] + 1
            print(f"[Resume] Loaded from epoch {_rs['epoch']} "
                  f"(val_loss={_rs['best_val_loss']:.4f}, patience={_rs['patience_counter']}/{PATIENCE})")
            print(f"[Resume] Continuing from epoch {start_epoch}")
        except Exception as _e:
            print(f"[Resume] Could not load state: {_e} — starting fresh")
            start_epoch = 1
    else:
        print("[Resume] No prior checkpoint found — starting fresh")

    _run_swap_baseline_gb = round(psutil.swap_memory().used / 1024**3, 3) if HAS_PSUTIL else 0.0
    if HAS_PSUTIL and device.type == "cpu":
        print(f"[RAM] Swap/page-file baseline for this run: {_run_swap_baseline_gb:.3f} GB "
              f"— every epoch below is checked against THIS fixed value, not a rolling one.")

    for epoch in range(start_epoch, 10**6):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()

        train_loss, train_acc, hw = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, device,
            epoch=epoch, img_size=img_size
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        current_lr = scheduler.get_lr()[0]
        elapsed    = time.time() - t0

        print(f"[{img_size}x{img_size}] Epoch {epoch:3d}  "
              f"loss: {train_loss:.4f}  acc: {train_acc:.4f}  |  "
              f"val_loss: {val_loss:.4f}  val_acc: {val_acc:.4f}  |  "
              f"lr: {current_lr:.2e}  [{elapsed:.0f}s]  |  "
              f"VRAM {hw['vram_peak_alloc_gb']:.1f}/{hw['vram_peak_reserved_gb']:.1f}GB  "
              f"CUDA {hw['cuda_util_pct']}%  {hw['gpu_temp_c']}°C  |  "
              f"CPU {hw['cpu_pct']}%  RAM {hw['ram_used_gb']:.1f}/{hw['ram_total_gb']:.1f}GB")

        for k, v in [("train_loss", train_loss), ("train_acc", train_acc),
                     ("val_loss", val_loss), ("val_acc", val_acc), ("lr", current_lr)]:
            history[k].append(v)
        history.setdefault("vram_peak_alloc_gb",    []).append(hw["vram_peak_alloc_gb"])
        history.setdefault("vram_peak_reserved_gb", []).append(hw["vram_peak_reserved_gb"])
        history.setdefault("cuda_util_pct",    []).append(hw["cuda_util_pct"])
        history.setdefault("gpu_temp_c",       []).append(hw["gpu_temp_c"])
        history.setdefault("cpu_pct",          []).append(hw["cpu_pct"])
        history.setdefault("ram_used_gb",      []).append(hw["ram_used_gb"])
        history.setdefault("epoch_time_s",     []).append(round(elapsed, 1))

        if device.type == "cpu" and HAS_PSUTIL:
            _swap_used_gb = round(psutil.swap_memory().used / 1024**3, 3)
            _free_ram_gb  = psutil.virtual_memory().available / 1024**3
            if _swap_used_gb > _run_swap_baseline_gb:
                print(f"  [RAM] Page file / swap usage GREW beyond this run's "
                      f"{_run_swap_baseline_gb:.3f} GB baseline ({_swap_used_gb:.3f} GB now) — "
                      f"treating as OOM, stopping training cleanly after epoch {epoch}")
                break
            if _free_ram_gb < RAM_RESERVE_GB:
                print(f"  [RAM] Free RAM ({_free_ram_gb:.2f} GB) below the "
                      f"{RAM_RESERVE_GB} GB reserve — stopping training "
                      f"cleanly after epoch {epoch}")
                break

        early_stop(val_loss, model)
        if early_stop.stop:
            break

        save_resume_state(
            cfg["resume_path"],
            epoch=epoch,
            patience_counter=early_stop.counter,
            best_val_loss=float(early_stop.best_loss),
            optimizer_state=optimizer.state_dict(),
            scheduler_state=scheduler.state_dict(),
            history=history,
        )

    print(f"\n[Train] [{img_size}x{img_size}] Loading best checkpoint...")
    ckpt = torch.load(cfg["checkpoint_path"], map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])

    print(f"\n[Eval] [{img_size}x{img_size}] Running per-class accuracy analysis on test set...")
    test_loss, test_acc = evaluate(model, test_loader, criterion, device, per_class=True)
    print(f"\n{'='*40}")
    print(f"  [{img_size}x{img_size}] SOAP Test accuracy : {test_acc:.4f}  ({test_acc*100:.2f}%)")
    print(f"  [{img_size}x{img_size}] SOAP Test loss     : {test_loss:.4f}")
    print(f"{'='*40}")

    plot_history(history, cfg["plot_path"], img_size, title="MNIST Digit Ensemble SOAP")
    save_log(history, cfg["log_path"])
    torch.save({"state_dict": model.state_dict()}, cfg["final_model_path"])
    print(f"[Save] {cfg['final_model_path']}")

    try:
        model_cpu = OCRConvNetTriplePyramid(num_classes=NUM_CLASSES, drop_path=DROP_PATH, dropout=DROPOUT_HEAD)
        model_cpu.load_state_dict(
            torch.load(cfg["final_model_path"], map_location="cpu", weights_only=False)["state_dict"]
        )
        export_onnx(model_cpu, cfg["onnx_path"], img_size, validate=True)
    except Exception as e:
        print(f"[ONNX] Export failed: {e}")

    clear_resume_state(cfg["resume_path"])
    print(f"\n[Done] [{img_size}x{img_size}] All files saved to {OUTPUT_ROOT}")
    return test_acc


# =============================================================================
# 6. MAIN
# =============================================================================

def main(batch_override=None):
    print("\n" + "#" * 60)
    print(f"  MNIST DIGIT ENSEMBLE — SOAP {IMG_SIZE}x{IMG_SIZE}")
    print(f"  Seed: GLOBAL_SEED={GLOBAL_SEED} (override with MNIST_SEED env var)")
    print("#" * 60 + "\n")

    acc = run_training(IMG_SIZE, batch_override=batch_override)

    print("\n" + "#" * 60)
    print("  TRAINING COMPLETE — SOAP")
    print(f"  {IMG_SIZE}x{IMG_SIZE}: {acc*100:.2f}% test accuracy")
    print("#" * 60)


if __name__ == "__main__":
    _parser = argparse.ArgumentParser()
    _parser.add_argument('--batch-size', type=int, default=None,
                         help='Override auto batch detection with a fixed batch size')
    _args = _parser.parse_args()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    sys.stdout = _Tee(OUTPUT_ROOT / f"v3_mnist_digit_soap_{IMG_SIZE}_cli_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    main(batch_override=_args.batch_size)
