# mnist-ocr-ensemble-v3

v3 of a multi-optimizer PyTorch OCR ensemble for handwritten digits.
Optimizers: SOAP, AdamW, Muon (Lion, SGD, AdaHessian dropped for weaker
real-world performance). Adds a case-classifying router (digit/
uppercase/lowercase/unknown) ahead of planned letter-reading models. No
post-processing — raw output only.

## Overview

This project trains a small ensemble of PyTorch CNNs to read handwritten
digits, evaluated against real-world accuracy rather than test-set
accuracy alone, on consumer-grade hardware (RTX 4080, no cloud compute).
v3 trains 15 digit models (3 optimizers — SOAP, AdamW, Muon — across 5
per-source resolution tiers, 16/28/32/64/128) plus a 5-resolution router
that classifies each detected character box as digit / uppercase /
lowercase / unknown before handing digit boxes to the ensemble. There is
no post-processing anywhere in this pipeline: if a model can't read
something, that's the result.

## What changed in v3

Full detail, reasoning, and a per-model settings table are in
[`v3_CHANGELOG.md`](./v3_CHANGELOG.md) — this is a pointer, not a duplicate.

- **Optimizer roster:** dropped Lion (reproducible ~20-point real-world
  accuracy gap at 128×128), SGD (consistently weakest performer), and
  AdaHessian (already out of rotation — excessive compute cost). Kept
  SOAP and AdamW. Added Muon, with an AdamW fallback for the non-2D
  parameters (biases, norm layers) Muon isn't designed for.
- **Resolution tiers now split by dataset source:** MNIST/EMNIST
  Digits/SVHN/ARDIS IV ladder 28→32→64→128; USPS ladders 16→32→64→128
  (its own native resolution, and the only source at 16×16 — that tier
  gets its own dedicated USPS-only model, not just a skip; see
  `v3_CHANGELOG.md`). Both ladders converge at 32/64/128, where every
  source is included.
- **Router added:** the v2 binary digit/not-digit gate (AdaBelief) is
  replaced — not run alongside — by a 4-verdict router (digit / UC / LC
  / `[UNK]`, Ranger optimizer) ahead of a planned, not-yet-built letter-
  reading phase. `[UNK]` is a confidence-threshold call, not a trained
  class.
- **Modularized (infrastructure only):** batch-size auto-detection,
  checkpoint/resume, telemetry, seeding, CLI logging, ONNX export, mixed
  precision, and LR scheduling all moved into a shared `common/` package,
  imported by all 20 training scripts instead of each carrying its own
  copy. Optimizer *algorithms* (Muon, Ranger) are deliberately **not**
  shared modules — each lives inline in the training script(s) that use
  it, per the task's own instruction and Part 2's modularization scope
  (which never included optimizer implementations) — see `v3_CHANGELOG.md`.
- **Renamed:** every model file now follows
  `v3_mnist_digit_{optimizer}_{resolution}` /
  `v3_mnist_router_ranger_{resolution}`.

## Repository structure

The tree below is the full projected layout once every model has been
trained and exported — right after cloning, only the `.py` files, this
README, and `v3_CHANGELOG.md` exist; every directory below a training
script (checkpoints, ONNX exports, logs, plots) is created by that
script the first time it runs.

```
mnist-ocr-ensemble-v3/
├── README.md
├── v3_CHANGELOG.md
├── requirements.txt                     # pip install -r requirements.txt
├── setup_packages.py                    # python setup_packages.py [--check]
├── run_all_training.ps1                 # runs all 20 training scripts in sequence, skips
│                                         # already-completed ones — see the script's own header
│
├── common/                              # shared INFRASTRUCTURE modules (Part 2 modularization —
│   │                                     # optimizer algorithms are NOT here, see below)
│   ├── __init__.py
│   ├── seeding.py                       # GLOBAL_SEED / set_all_seeds / CPU thread reservation
│   ├── telemetry.py                     # nvidia-smi + psutil hardware stats (power/clocks/
│   │                                     # throttle/mem-util/fan/disk I/O, min/avg/max per epoch)
│   ├── batch_sizing.py                  # auto batch-size detection (bottom-up + binary search)
│   ├── checkpointing.py                 # EarlyStopping, resume save/clear helpers
│   ├── cli_logging.py                   # _Tee transcript mirror, CSV log, training-curve plot
│   ├── onnx_export.py                   # export_onnx()
│   ├── scheduler.py                     # WarmupCosineScheduler
│   ├── amp.py                           # mixed-precision train step (autocast/GradScaler/clip)
│   └── distributed.py                   # optional multi-GPU DDP support — NOT YET VALIDATED
│                                         # against real multi-GPU hardware, see its own docstring
│
├── supplementary_data.py                # shared dataset loader (extends v2's own module)
├── ocr_pipeline_mnist.py                # ONNX inference pipeline (router + digit ensemble)
│                                         # NOTE: Muon and Ranger have NO standalone file — each
│                                         # optimizer is implemented directly inside the training
│                                         # script(s) that use it (v3_mnist_digit_muon_*.py /
│                                         # v3_mnist_router_ranger_*.py), not a shared module —
│                                         # see v3_CHANGELOG.md.
│
├── v3_mnist_digit_soap_16.py             # ── SOAP digit models (5 resolutions) ──
├── v3_mnist_digit_soap_16/                # 16x16 = USPS only (load_base_usps())
│   ├── v3_mnist_digit_soap_16_best.pt
│   ├── v3_mnist_digit_soap_16_final.pt
│   ├── v3_mnist_digit_soap_16.onnx
│   ├── v3_mnist_digit_soap_16_log.csv
│   ├── v3_mnist_digit_soap_16_curves.png
│   └── v3_mnist_digit_soap_16_cli_<timestamp>.txt
├── v3_mnist_digit_soap_28.py
├── v3_mnist_digit_soap_28/               # (same file set as above, _28 suffix)
├── v3_mnist_digit_soap_32.py
├── v3_mnist_digit_soap_32/               # (same file set as above, _32 suffix)
├── v3_mnist_digit_soap_64.py
├── v3_mnist_digit_soap_64/               # (same file set as above, _64 suffix)
├── v3_mnist_digit_soap_128.py
├── v3_mnist_digit_soap_128/              # (same file set as above, _128 suffix)
│
├── v3_mnist_digit_adamw_16.py            # ── AdamW digit models (5 resolutions) ──
├── v3_mnist_digit_adamw_16/               # 16x16 = USPS only
├── v3_mnist_digit_adamw_28.py
├── v3_mnist_digit_adamw_28/
├── v3_mnist_digit_adamw_32.py
├── v3_mnist_digit_adamw_32/
├── v3_mnist_digit_adamw_64.py
├── v3_mnist_digit_adamw_64/
├── v3_mnist_digit_adamw_128.py
├── v3_mnist_digit_adamw_128/
│
├── v3_mnist_digit_muon_16.py             # ── Muon digit models (5 resolutions) ──
├── v3_mnist_digit_muon_16/                # 16x16 = USPS only
├── v3_mnist_digit_muon_28.py
├── v3_mnist_digit_muon_28/
├── v3_mnist_digit_muon_32.py
├── v3_mnist_digit_muon_32/
├── v3_mnist_digit_muon_64.py
├── v3_mnist_digit_muon_64/
├── v3_mnist_digit_muon_128.py
├── v3_mnist_digit_muon_128/
│
├── v3_mnist_router_ranger_16.py          # ── Router models (5 resolutions) ──
├── v3_mnist_router_ranger_16/             # 16x16 digit class = USPS only
│   ├── v3_mnist_router_ranger_16_best.pt
│   ├── v3_mnist_router_ranger_16_final.pt
│   ├── v3_mnist_router_ranger_16.onnx
│   ├── v3_mnist_router_ranger_16_log.csv
│   ├── v3_mnist_router_ranger_16_curves.png
│   └── v3_mnist_router_ranger_16_cli_<timestamp>.txt
├── v3_mnist_router_ranger_28.py
├── v3_mnist_router_ranger_28/
├── v3_mnist_router_ranger_32.py
├── v3_mnist_router_ranger_32/
├── v3_mnist_router_ranger_64.py
├── v3_mnist_router_ranger_64/
├── v3_mnist_router_ranger_128.py
├── v3_mnist_router_ranger_128/
│
└── pipeline_logs/                       # ocr_pipeline_mnist.py's own per-run _Tee transcripts
    └── ocr_mnist_<timestamp>.log
```

### Current repo contents — what's tracked in git (as of this writing)

The tree above is the full *projected* layout once every model has been
trained. Below is what's actually tracked in git and will appear in the
GitHub repo — confirmed against a recursive local listing, then
filtered through `.gitignore`. **9 of the 20 models have completed a
full training run** (ONNX export, log, curves, and CLI transcript all
tracked); **2 are mid-run** (only a CLI transcript is tracked so far —
no ONNX export exists yet); **1 has an empty output directory** (a
prior run attempt, since cleared); the remaining **8 have never been
run at all** — no directory exists for them yet:

```
mnist-ocr-ensemble-v3/
├── .gitignore
├── CLAUDE.md                          # Claude Code operating rules for this repo
├── README.md
├── v3_CHANGELOG.md
├── requirements.txt
├── setup_packages.py
├── run_all_training.ps1               # runs all 20 scripts in sequence, skips completed ones
├── ocr_pipeline_mnist.py
├── supplementary_data.py
│
├── common/
│   ├── __init__.py
│   ├── amp.py
│   ├── batch_sizing.py
│   ├── checkpointing.py
│   ├── cli_logging.py
│   ├── distributed.py
│   ├── onnx_export.py
│   ├── scheduler.py
│   ├── seeding.py
│   └── telemetry.py
│
├── historical_data/
│   └── claude_code_prompt.md          # archived prompt from an earlier session — kept for reference only
│
├── v3_mnist_digit_soap_16.py           # ── COMPLETE — full run + ONNX export ──
├── v3_mnist_digit_soap_16/
│   ├── v3_mnist_digit_soap_16.onnx
│   ├── v3_mnist_digit_soap_16_log.csv
│   ├── v3_mnist_digit_soap_16_curves.png
│   └── v3_mnist_digit_soap_16_cli_20260728_025105.txt
├── v3_mnist_digit_soap_28.py           # ── COMPLETE — full run + ONNX export (2 CLI transcripts — stopped and resumed) ──
├── v3_mnist_digit_soap_28/
│   ├── v3_mnist_digit_soap_28.onnx
│   ├── v3_mnist_digit_soap_28_log.csv
│   ├── v3_mnist_digit_soap_28_curves.png
│   ├── v3_mnist_digit_soap_28_cli_20260728_030120.txt
│   └── v3_mnist_digit_soap_28_cli_20260728_045546.txt
├── v3_mnist_digit_soap_32.py           # ── IN PROGRESS locally — no ONNX export yet ──
├── v3_mnist_digit_soap_32/              # .pt checkpoint + resume file exist locally, gitignored
│   └── v3_mnist_digit_soap_32_cli_20260729_031303.txt
├── v3_mnist_digit_soap_64.py           # ── run attempted, output cleared — empty dir exists ──
├── v3_mnist_digit_soap_64/              # (empty)
├── v3_mnist_digit_soap_128.py           # ── never run — source only, no directory ──
│
├── v3_mnist_digit_adamw_16.py          # ── COMPLETE — full run + ONNX export ──
├── v3_mnist_digit_adamw_16/              # (same file set as soap_16 above, adamw_16 naming)
├── v3_mnist_digit_adamw_28.py          # ── COMPLETE — full run + ONNX export (2 CLI transcripts — stopped and resumed) ──
├── v3_mnist_digit_adamw_28/
│   ├── v3_mnist_digit_adamw_28.onnx
│   ├── v3_mnist_digit_adamw_28_log.csv
│   ├── v3_mnist_digit_adamw_28_curves.png
│   ├── v3_mnist_digit_adamw_28_cli_20260728_053717.txt
│   └── v3_mnist_digit_adamw_28_cli_20260728_220102.txt
├── v3_mnist_digit_adamw_32.py           # ── never run — source only, no directory ──
├── v3_mnist_digit_adamw_64.py
├── v3_mnist_digit_adamw_128.py
│
├── v3_mnist_digit_muon_16.py           # ── COMPLETE — full run + ONNX export ──
├── v3_mnist_digit_muon_16/               # (same file set as soap_16 above, muon_16 naming)
├── v3_mnist_digit_muon_28.py           # ── COMPLETE — full run + ONNX export ──
├── v3_mnist_digit_muon_28/               # (same file set as soap_16 above, muon_28 naming)
├── v3_mnist_digit_muon_32.py            # ── never run — source only, no directory ──
├── v3_mnist_digit_muon_64.py
├── v3_mnist_digit_muon_128.py
│
├── v3_mnist_router_ranger_16.py        # ── COMPLETE — full run + ONNX export ──
├── v3_mnist_router_ranger_16/            # (same file set as soap_16 above, router_ranger_16 naming)
├── v3_mnist_router_ranger_28.py        # ── COMPLETE — full run + ONNX export ──
├── v3_mnist_router_ranger_28/            # (same file set as soap_16 above, router_ranger_28 naming)
├── v3_mnist_router_ranger_32.py        # ── COMPLETE — full run + ONNX export ──
├── v3_mnist_router_ranger_32/            # (same file set as soap_16 above, router_ranger_32 naming)
├── v3_mnist_router_ranger_64.py        # ── IN PROGRESS locally — no ONNX export yet ──
├── v3_mnist_router_ranger_64/           # .pt checkpoint + resume file exist locally, gitignored
│   └── v3_mnist_router_ranger_64_cli_20260728_213658.txt
└── v3_mnist_router_ranger_128.py        # ── never run — source only, no directory ──
```

`__pycache__/` directories under the repo root and under `common/`,
`.claude/settings.local.json`, `.vscode/settings.json` (Claude Code's
and VS Code's own local session/workspace settings, not project
content), and every `_best.pt` / `_final.pt` / `_resume.pt` checkpoint
file (present locally for every trained or mid-run model, but not
needed to run inference — `ocr_pipeline_mnist.py` loads the `.onnx`
export, never the `.pt` checkpoint) are excluded from this tree and
from the repo itself — see `.gitignore` for the exact rules.

### External data locations (not part of the repo)

`supplementary_data.py`'s dataset path constants point at a specific
location on the original project machine, **outside this repo entirely**
(a different drive letter on Windows). They are kept exactly as they
were rather than re-pointed, per this project's own convention — see the
"EDIT THESE FOR YOUR OWN SYSTEM" comment block directly above each
constant in `supplementary_data.py`. **This is a known, accepted
limitation of a single-user research project, not an oversight:** none
of the paths below exist on a fresh clone, and every training/router
script will fail to load data until you edit `supplementary_data.py`
(and each training script's own local `DATA_DIR` copy — see
`v3_CHANGELOG.md`'s modularization note on why each script still carries
one) to point at wherever you keep these datasets.

```
E:\CSC-114\emnist-model\datasets\               (DATA_DIR's parent — original machine)
├── pytorch\            # DATA_DIR — MNIST, EMNIST Digits, EMNIST Balanced, USPS, SVHN
│                        # (all torchvision-managed, auto-download here on first use)
├── ardis\               # ARDIS_DIR — ardis_images.npy / ardis_labels.npy
│                        # (auto-downloaded and prepared on first use if missing)
├── kaggle\               # KAGGLE_DIR — optional, unused by default (az_images.npy / az_labels.npy)
├── EnglishHnd\English\Hnd\Img\   # CHARS74K_HND — optional, unused by default
├── EnglishImg\English\Img\GoodImg\Bmp\  # CHARS74K_IMG — optional, unused by default
└── pg_hwld\              # PGHWLD_DIR — optional, unused by default
```

## Usage

Full CLI usage for each script is documented in its own module
docstring — this section is pointers, not a manual.

**Train one digit model** (auto-detects batch size, resumes automatically
if an interrupted run's resume file is present):
```
python v3_mnist_digit_soap_64.py
python v3_mnist_digit_adamw_128.py --batch-size 512   # override auto-detection
```

**Train one router model:**
```
python v3_mnist_router_ranger_64.py
python v3_mnist_router_ranger_64.py --infer some_letter.png   # spot-check a trained checkpoint
```

**Multi-GPU:** two independent modes — see `v3_CHANGELOG.md`'s multi-GPU
entry for full detail and the honest caveat on which mode has been
tested where.

- **Run different models on different cards at once** (any script,
  `--gpu N` selects a physical GPU index; default uses whatever CUDA
  already considers current):
  ```
  python v3_mnist_digit_soap_64.py --gpu 0     # terminal 1
  python v3_mnist_digit_muon_64.py --gpu 1     # terminal 2, at the same time
  ```
- **Split ONE model's training across multiple GPUs** (DistributedDataParallel,
  launched via `torchrun`, one process per GPU). **⚠️ Not yet validated
  against real multi-GPU hardware** — built and reasoned through
  carefully (see `common/distributed.py`'s own docstring and
  `v3_CHANGELOG.md`'s multi-GPU entry), but every real number anywhere
  in this project came from a single GPU; confirm on real hardware
  before trusting it for an actual multi-day run:
  ```
  torchrun --standalone --nproc_per_node=2 v3_mnist_digit_soap_64.py
  ```
  Do not also pass `--gpu` under `torchrun` — `LOCAL_RANK` (set by the
  launcher) already picks each process's device correctly.

**Run the inference pipeline** — single model, ensemble, or a whole
directory of `.onnx` files, with or without the router pre-filter (see
`ocr_pipeline_mnist.py`'s own module docstring for the full flag list
and more examples):
```
python ocr_pipeline_mnist.py --models v3_mnist_digit_soap_64/v3_mnist_digit_soap_64.onnx page.jpg

python ocr_pipeline_mnist.py --model-dir . page.jpg   # every .onnx found recursively

python ocr_pipeline_mnist.py \
    --router-model v3_mnist_router_ranger_64/v3_mnist_router_ranger_64.onnx \
    --model-dir . page.jpg
```

Every environment variable / override this project's scripts read (seed
override via `MNIST_SEED`, dataset root override via `ROUTER_DATA_DIR`,
etc.) is documented at its point of use in the relevant script's own
docstring or argparse `--help` output.

## Requirements

Verified against actual imports across all v3 files (not guessed):

```
torch
torchvision
numpy
matplotlib
opencv-python        # cv2 — ocr_pipeline_mnist.py only
Pillow                # PIL — supplementary_data.py, router --infer
psutil                # hardware telemetry (common/telemetry.py, common/batch_sizing.py)
onnx                  # SOAP scripts' export validation (common/onnx_export.py's validate=True path)
onnxruntime-gpu        # ocr_pipeline_mnist.py inference (NOT plain onnxruntime — that's CPU-only)
pytorch_optimizer      # SOAP — v3_mnist_digit_soap_*.py
schedulefree           # Schedule-Free AdamW — v3_mnist_digit_adamw_*.py
```

Muon and Ranger are implemented directly inline in the training scripts
that use them (`v3_mnist_digit_muon_*.py`, `v3_mnist_router_ranger_*.py`
— see the Repository structure note above) — no separate optimizer
package needed for either, and no standalone project module either.
`rarfile` is an optional fallback dependency for `supplementary_data.py`'s
ARDIS auto-download (only used if none of 7-Zip/`unrar`/`unar` are found
on `PATH`).

**To install:** `requirements.txt` lists the same set above for a plain
`pip install -r requirements.txt` (after installing `torch`/`torchvision`
from the CUDA-specific wheel index — see the file's own header comment).
Or run `python setup_packages.py`, which handles the CUDA wheel index
itself and has a `--check` mode that reports what's installed/missing
without installing anything:
```
python setup_packages.py --check   # verify only, installs nothing
python setup_packages.py           # install everything
```
