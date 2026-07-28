# Changelog — v3 Restructure (Optimizer Roster, Modularization, Router)

Project: MNIST OCR ensemble, v3. This is a new document — it does not
extend the v2 repo's own `v3_CHANGELOG.md` (a separate, historical
reference, left untouched and not copied from). Every file this entry
describes is a v3 output file, adapted from a named v2 counterpart; see
the "File change list" section at the bottom for the full adapted-from
mapping.

---

## 2026-07-27 — v3 restructure: optimizer roster, modularization, router

### Problem / context

The v2 project trained 4 optimizers (AdamW, Lion, SGD, SOAP) at 3
resolutions each (32/64/128), plus a standalone binary digit/not-digit
gate model (AdaBelief). AdaHessian had already been dropped from active
training earlier in the project's history (24+ hour runs at 128×128 with
no wall-clock limit — too expensive to keep in rotation) but its scripts
were still present in the repo. The project's goal is real-world
digit-reading accuracy, not test-set novelty, and two of the four active
optimizers had known, documented weaknesses for that goal: Lion showed a
reproducible ~20-point real-world accuracy gap at 128×128 in prior
validation (independently confirmed on a second machine), and SGD was
consistently the weakest performer across every resolution tested. This
restructure (1) changes the optimizer roster to match a trustworthy-
ensemble goal rather than a documented-failure-case goal, (2) pulls the
per-script duplicated infrastructure (batch-sizing, checkpoint/resume,
telemetry, seeding, CLI logging, ONNX export) into shared modules so 16
training scripts stop carrying 16 near-identical copies of the same
code, and (3) expands the v2 binary gate into a 4-verdict router
(digit / UC / LC / [UNK]) ahead of a planned (not yet built) letter-
reading phase.

### What was verified before changing anything

Before writing any v3 file, the actual v2 repo was read directly (not
relied on from prior discussion): `supplementary_data.py` (919 lines),
`mnist_digit_gate.py` (1376 lines) in full, `ocr_pipeline_mnist.py`
(1503 lines) in full, `mnist_soap_64.py` and `mnist_adamw_64.py` in full
as representative training-script templates, `mnist_sgd_64.py`'s and
`mnist_adamw_64.py`'s architecture sections, `v3_CHANGELOG.md` and
`README.md` for v2's own documentation conventions, and `setup_packages.py`
for the actual dependency list. Specifically checked, not assumed:

- **That AdamW is actually Schedule-Free AdamW**, not vanilla AdamW —
  confirmed via `mnist_adamw_64.py`'s own `schedulefree.AdamWScheduleFree`
  import and its `optimizer.train()`/`optimizer.eval()` + 50-batch
  BatchNorm-warmup toggle pattern. This is preserved exactly in v3 — see
  "Modularization" below for what could and could not be safely
  extracted around it.
- **That each optimizer has its own dedicated architecture**, not a
  shared one — confirmed by reading each script's own architecture
  section: AdamW → `OCRConvNetWide` (32→128→256→512, SE attention,
  StochasticDepth, ~9.7M params), SOAP → `OCRConvNetTriplePyramid`
  (96→192→384→768 feature pyramid, ~4.6M params), SGD →
  `OCRConvNetTripleSGD`, Lion → `OCRConvNet`, AdaHessian →
  `OCRConvNetTripleAdaHessian`. Diffed `mnist_adamw_32.py` against
  `mnist_adamw_64.py` (and `mnist_soap_32/64/128.py` against each other)
  directly: confirmed the only differences between resolution variants
  of the same optimizer are the resolution number itself and its
  downstream file paths — every hyperparameter constant (PATIENCE, LR,
  betas, weight decay, etc.) is identical across resolutions for a given
  optimizer. This is what made a template-based generation approach for
  the digit scripts (ultimately 15 — see "Resolution ladder split" below
  for the 16×16-tier correction) safe (see "Modularization" below)
  rather than risking silently losing a resolution-specific tuning value
  that turned out not to exist.
- **That `mnist_digit_gate.py` is a 2-class binary gate**, not a
  4-class router — confirmed its `LABEL_MAP = ["??", "digit"]`,
  `NUM_CLASSES = 2`, and `OCRGateNet`'s `Linear(32, num_classes)` output
  head. Confirmed via `grep` across the repo that no more advanced
  router implementation exists anywhere else in v2.
- **Every reference to `GATE-NOT-DIGIT`, `gate_verdict`, `gate_prob`,
  and `--gate-model`** in `ocr_pipeline_mnist.py` was located and read in
  context before any router wiring was written — see "Router
  wiring decision" below for what was found and how each site was
  changed.
- **That `BALANCED_TO_BYCLASS` in `supplementary_data.py` only mapped
  byclass indices 0-35** (digits + uppercase), not the full 0-61 —
  confirmed by reading the dict literal directly. See "Bugs found while
  re-enabling EMNIST Balanced" below.
- **That no int8 quantization code exists anywhere in the v2 codebase** —
  `grep`'d for `quantiz`/`int8` across every file; the only hits were a
  single docstring line in `mnist_adamw_64.py`/`mnist_lion_*.py` citing a
  textbook chapter that *covers* quantization as a topic, with no actual
  quantization call anywhere. Nothing was modularized for this in
  `common/` as a result — there was no duplicated logic to extract. Noted
  explicitly here rather than silently ignored, since Part 2 of this
  restructure's task description named int8 quantization as one of the
  things to modularize.

### Optimizer roster change

**Dropped:** Lion, SGD, AdaHessian. **Kept:** SOAP, AdamW. **Added:** Muon.

- Lion and SGD: dropped for the real-world accuracy reasons in
  "Problem/context" above — a trustworthy ensemble was prioritized over
  documenting an interesting failure case.
- AdaHessian: was already out of active rotation before this restructure
  (excessive compute cost); this restructure removes its scripts,
  checkpoints, and every reference to it from the active codebase
  entirely, per the task's explicit instruction, rather than leaving
  unused files behind.
- Muon: added based on 2025-2026 independent benchmarks showing it
  reliably outperforms AdamW-class optimizers across multiple domains at
  lower per-step compute cost than SOAP. Muon is defined only for 2D+
  parameters (weight matrices) — implemented directly inside each
  `v3_mnist_digit_muon_*.py` file (not a shared module — see "Naming and
  modularization-scope correction" below) as the standard
  `MuonWithAuxAdam`-style pattern (a single `torch.optim.Optimizer`
  with two param_group "flavors": Newton-Schulz-orthogonalized momentum
  for `ndim >= 2` parameters, standard decoupled-weight-decay AdamW for
  everything else — biases, BatchNorm weight/bias), matching the task's
  explicit requirement for an AdamW fallback on non-2D parameters. The
  Newton-Schulz quintic-iteration coefficients (3.4445, -4.7750, 2.0315)
  are the published values from Jordan et al.'s Muon release; this
  project's implementation runs the iteration in float32 throughout
  (not bfloat16, unlike some reference implementations) so behavior is
  identical whether a run falls back to CPU or not, at some extra
  compute cost on GPU versus a bf16 version — a deliberate, documented
  trade-off, not an oversight.

**No optimizer received carried-over trained weights** — every v3 model
(SOAP, AdamW, Muon digit models; Ranger router) trains from scratch, per
the task's explicit constraint.

### Resolution ladder split (28-start vs. 16-start)

MNIST, EMNIST Digits, SVHN, and ARDIS IV ladder 28→32→64→128; USPS
ladders 16→32→64→128 (USPS's own native resolution, since 28×28 is
neither native nor meaningfully closer to native for USPS than either of
its real neighbors). Both ladders converge at 32/64/128, where every
source is included — only the bottom tier differs.

**Correction (post-delivery, per direct user follow-up): 16×16 is its
own separate USPS-only model, not skipped.** The first pass at this
restructure read the task's own arithmetic ("3 optimizers × 4 resolution
tiers = 12 digit models total") as meaning there is only ONE bottom tier
(28×28), with USPS simply excluded from it and no 16×16 model trained at
all — that reading is what the "12"/"4 tiers" count requires taken in
isolation, but it directly contradicts another line in the same task
text: "...since USPS is absent from the 28×28 tier **and everyone else
is absent from the 16×16 tier**" — which only makes sense if a 16×16
tier, with its own (USPS-only) training data, actually exists. That
sentence was under-weighted in the first pass. Asked directly, the user
confirmed: yes, USPS gets its own dedicated 16×16 model (and router),
alongside the 28×28 model everyone else uses. **There are 5 resolution
tiers, not 4** — 16/28/32/64/128 — meaning **15 digit models** (3
optimizers × 5 tiers) and **5 router models**, not 12 and 4. The "12
digit models total" / "4 resolution tiers" line in the original task
text is treated as the error here, superseded by the corrected count
below and by the user's direct confirmation.

This has a real data-pipeline consequence beyond just adding more
`.py` files: at every other tier, base MNIST (`load_base_mnist()`) is
the training "spine" (source of the train/val/test split), with
`load_supplementary()` adding other sources on top. At 16×16, USPS IS
the entire spine — there's no MNIST involved at all, since USPS is the
only source in that tier's ladder. This needed a new function,
`supplementary_data.load_base_usps()`, mirroring `load_base_mnist()`'s
structure exactly (same 85/15 split, same `GLOBAL_SEED`-derived
`split_seed`, same `return_train_targets` escape hatch) but built on
`USPSDataset` instead of raw `torchvision.datasets.MNIST`. Every digit
training script's `load_mnist()` and the router's `load_router_data()`
now branch — `base_loader = load_base_usps if img_size == 16 else
load_base_mnist` — before calling into whichever one matches the
resolution tier; everything downstream (weighting, dataloader
construction) is unchanged, since both loaders return the same
`(train_ds, val_ds, test_ds[, train_targets])` shape.

`supplementary_data.digit_sources_for_tier(img_size)` is the one
function that encodes the whole split, called by every v3 digit script
and every router script instead of each one hardcoding its own
`use_usps=...` flag: at 16×16 it returns every flag `False` (there's
nothing left for `load_supplementary()` to add — USPS is already the
entire spine via `load_base_usps()`; calling it anyway is harmless, it
just prints "no supplementary data available" and returns `None`); at
28×28, every source except USPS; at 32/64/128, every source. 16/28/32/
64/128 are the only valid inputs (raises `ValueError` otherwise), and
it's the single place this split is encoded, so every consumer stays
consistent with it by construction rather than by each one re-deriving
the rule.

### Modularization (Part 2)

New `common/` package, imported by every v3 training script instead of
each carrying its own copy:

- `common/seeding.py` — `apply_cublas_workspace_config()` /
  `get_global_seed()` / `set_all_seeds()` / `reserve_cpu_threads()`.
  Byte-identical logic to the block that was duplicated in all 16 v2
  training scripts; only the code's location moved, and the previously-
  inline sequence is now four explicit function calls in the same
  required order (documented in the module docstring, since getting the
  order wrong silently reintroduces nondeterminism).
- `common/telemetry.py` — `get_hw_stats()` / `print_vram_baseline()` /
  `setup_device()`. Same nvidia-smi/psutil-based telemetry, byte-
  identical detection logic.
- `common/batch_sizing.py` — `determine_batch_size()` /
  `_probe_batch_size()`. This is the one module where "identical logic"
  required real parameterization to extract safely: the v2 scripts'
  probe functions differed only in whether the probe step used
  AMP+GradScaler+`clip_grad_norm_(max_norm=1.0)` (AdamW, the v2 gate) or
  a plain float32 backward with neither (SOAP, since Kronecker
  eigendecomposition requires float32). The shared function now takes
  `use_amp` and `grad_clip_norm` parameters and both original code paths
  are still present, byte-for-byte, selected by those parameters —
  nothing about the OOM/shared-VRAM/CPU-RAM/swap detection logic changed.
  Muon and the router both use the AMP+clip path (see their own
  standard-settings entries below for why).
- `common/checkpointing.py` — `EarlyStopping` (byte-identical) plus
  `save_resume_state()`/`clear_resume_state()` wrapping the try/except
  torch.save() pattern every v2 script duplicated. Explicitly NOT
  generalized: the "does the saved batch size match this run's
  auto-detected batch size" resume-compatibility check (present in the
  v2 gate, and now in the v3 router and Muon/SOAP scripts) stays in each
  training script's own resume block, since which state pieces are
  batch-size-dependent differs by optimizer (SOAP/Muon: optimizer +
  scheduler state; AdamW: optimizer + scaler state, no scheduler; the
  router: optimizer + scheduler state) — generalizing that check would
  have meant either a lowest-common-denominator interface that hides
  which fields it's actually checking, or an overly clever one that's
  harder to audit than just reading each script's own resume block.
- `common/cli_logging.py` — `_Tee` (append-mode, timestamped session
  header — the training-script variant, not `ocr_pipeline_mnist.py`'s
  own per-run-file `_Tee`, which stayed local to that file since its
  behavior is genuinely different), `plot_history()` (title now a
  parameter instead of a hardcoded per-optimizer string), `save_log()`
  (identical CSV fieldnames/format).
- `common/onnx_export.py` — `export_onnx()`, with an added `validate`
  parameter (the SOAP scripts' extra `onnx.checker.check_model()` round-
  trip, optional so scripts that don't otherwise need the `onnx` package
  don't gain a new dependency just for this).
- `common/scheduler.py` — `WarmupCosineScheduler`, extracted from the
  identical class duplicated in the v2 SOAP scripts, AdaHessian scripts,
  and the digit gate. Works unchanged over any number of param_groups
  (each group's own base LR, captured at construction, is scaled by the
  same curve) — this is what lets Muon's optimizer (two param_groups at
  different base LRs: the Muon group and its AdamW-fallback group) use
  it with no extra code.

`supplementary_data.py` — extended, not replaced, per the task's
explicit instruction: the existing per-source `Dataset` wrapper classes,
`load_supplementary()`, and `load_base_mnist()` are unchanged in
behavior. Three additions:
1. `digit_sources_for_tier(img_size)` — see "Resolution ladder split"
   above.
2. `load_base_usps()` — the 16×16 tier's USPS-only equivalent of
   `load_base_mnist()`, added after the resolution-tier correction
   above.
3. `BALANCED_TO_BYCLASS` fix — see "Bugs found" below.

**Preserved exactly, not touched by this modularization pass:** auto
batch-size detection's actual candidate ladder and refinement algorithm,
checkpoint/resume semantics, telemetry sampling behavior, the 85/15
`load_base_mnist()` train/val split methodology (still seed=42 via
`GLOBAL_SEED`), and every existing architecture (`OCRConvNetWide` for
AdamW, `OCRConvNetTriplePyramid` for SOAP) — Part 2 of this restructure
was scoped as structural only, and no training-behavior changes were
made to the two optimizers being kept.

**Not modularized, and why:** model architectures stay one-per-script
(matching the existing v2 convention — the task's Part 2 list of things
to modularize did not include architectures, and unifying them would
have been a design change beyond "structural refactor, not a behavior
change"). Schedule-Free AdamW's `optimizer.train()`/`optimizer.eval()` +
BatchNorm-warmup toggle sequence also stays inline in
`v3_mnist_digit_adamw_*.py` rather than becoming a shared helper — it's
only used by one optimizer, and wrapping a 4-line, order-sensitive
sequence in a function across a 2-line call site would have added
indirection without reducing real duplication (there is only one caller).

### Naming and modularization-scope correction (post-delivery, per direct user follow-up)

Two real mistakes in the first delivered pass, both caught by direct
user follow-up, not found independently:

**1. Muon and Ranger were wrongly factored into standalone project
modules (`muon_optimizer.py`, `ranger_optimizer.py`) — reverted.** The
task text says explicitly, about Muon's AdamW fallback: *"Use AdamW as
that fallback **within the Muon training script**"* — the optimizer
implementation was always meant to live inside the training script(s)
that use it, not in a project-wide shared file. Part 2's own
modularization list (auto batch-size detection, checkpoint/resume,
telemetry, dataset loading, mixed precision, ONNX export, int8
quantization, seed/determinism) never included optimizer algorithms —
only infrastructure. Creating those two files was scope not asked for.
Reverted: `zeropower_via_newtonschulz5()` and the `Muon` class now live
directly inside each `v3_mnist_digit_muon_{16,28,32,64,128}.py`; `RAdam`
and `Lookahead` now live directly inside each
`v3_mnist_router_ranger_{16,28,32,64,128}.py`. Both standalone files were
deleted; nothing else imports them. This also resolves a secondary
complaint that was really the same root cause, not a separate one — the
two files not carrying a `v3_` prefix — since once the code lives inside
the already-correctly-named training scripts, there's no separate file
left to name.

**2. "Mixed precision setup" — explicitly named in Part 2's
modularization list — had not actually been modularized.** Every AdamW/
Muon/router script was still building its own `GradScaler` and repeating
the same `autocast` → `scale().backward()` → `unscale_()` →
`clip_grad_norm_()` → `step()` → `update()` sequence inline. Fixed:
`common/amp.py` (new) — `amp_train_step()` runs one full AMP forward/
backward/step and returns whether GradScaler actually stepped the
optimizer (so callers with a per-step scheduler know whether to advance
it); `build_grad_scaler()` constructs the `GradScaler`. Imported by
`v3_mnist_digit_adamw_*.py`, `v3_mnist_digit_muon_*.py`,
`v3_mnist_router_ranger_*.py`, and by `common/batch_sizing.py`'s own
probe (which now calls the same function instead of repeating a close
copy of the same mechanics a third time — the probe and the real
training loop share the literal same AMP step code, not just similar
code). SOAP does not import this module — SOAP never uses AMP (float32
required for its Kronecker eigendecomposition), so there is nothing for
it to import.

**Also fixed while addressing the above (a separate, smaller item raised
in the same follow-up round):** `OUTPUT_ROOT` in every training script
is script-relative and genuinely safe as-is on any machine — but the
task explicitly warns against *"silently assum[ing] script-relative
paths need no comment either... the person reading the code later needs
to know which paths are safe as-is versus which need personalizing."*
The first delivered pass added the "change this" comments on `DATA_DIR`
everywhere but left `OUTPUT_ROOT` uncommented, which is exactly the
silent-assumption the task said not to make. Fixed: every training
script's `OUTPUT_ROOT` now has an explicit comment confirming it's
script-relative and safe as-is, immediately above the line, matching the
treatment `DATA_DIR` already had.

**Resume/checkpoint implication of both fixes:** neither changes any
saved checkpoint's shape — `common/amp.py` only restructures *how* the
AMP step runs, not what state gets saved (`scaler.state_dict()` is
unaffected), and inlining Muon/Ranger doesn't change either optimizer's
own `state_dict()` structure (same `torch.optim.Optimizer`-based state,
same param_group layout) — only where the class definition physically
lives in the file tree. Both are safe to treat as pure refactors for
resume purposes, though moot anyway since no real checkpoint has been
produced yet (see "Verification" below).

### Bugs found while re-enabling EMNIST Balanced

`BalancedEMNISTDataset` (`supplementary_data.py`) filters samples via
`if label in BALANCED_TO_BYCLASS`. The v2 dict only had entries for
byclass 0-35 (digits + uppercase) — EMNIST Balanced's 11 EMNIST-Balanced-
distinct lowercase classes (byclass indices 36-46 in the dataset's own
label space: a, b, d, e, f, g, h, n, q, r, t, per the dataset's published
`emnist-balanced-mapping.txt`) were silently dropped by that filter, with
no error or warning. This had zero behavioral effect in v2 — `use_balanced`
was `False` at every call site, so `BalancedEMNISTDataset` was never
actually instantiated by any v2 script. It's a real bug all the same,
and it would have silently broken the v3 router's lowercase class had it
gone unnoticed (the router genuinely needs lowercase representation).
Fixed by extending `BALANCED_TO_BYCLASS` to also map indices 36-46 to
byclass 36-61 (`a`→36 through `t`→55, per the standard `byclass 36-61 = a-z`
convention `CHARS74K_TO_BYCLASS` already used) — verified independently
with a standalone Python snippet (no torch dependency) confirming all 47
EMNIST Balanced classes now map correctly, printed and checked by hand
against the known a/b/d/e/f/g/h/n/q/r/t list.

**Known, accepted, NOT fixed:** `load_supplementary(use_mnist=True)`
loads a second, full 60k-sample MNIST training copy as a "supplementary"
source, on top of the ~51k-sample train subset `load_base_mnist()`
already returns after its own 85/15 split — meaning some images that
landed in `load_base_mnist()`'s held-out validation subset can still
reach the model via that second copy during training. This predates v3;
Part 2 of this restructure is a structural refactor only (no behavior
changes to preserved logic), so it was left exactly as it was in v2 —
flagged explicitly here (and in `supplementary_data.py`'s own docstring)
rather than silently carried forward unmentioned.

### Router wiring decision (gate replacement — resolved explicitly)

The task's own text contains two adjacent instructions that are in real
tension if read too literally: one section says the marker-collision
resolution keeps "`??`'s existing two meanings unchanged (ensemble split
/ gate not-digit)," while the very next section explicitly asks for a
decision on whether the router *replaces* the gate or runs *alongside*
it. Read together, the "keeps existing two meanings" text describes the
**pre-restructure** state (used to establish why `[UNK]` needs its own
marker rather than overloading `??` a third way) — it does not, on its
own, mandate that both `??` meanings must remain live after the
replace-vs-alongside decision is made.

**Resolved: the router fully replaces the gate.** `--gate-model`,
`predict_gate()`, `GATE_UNSURE_LOW`/`GATE_UNSURE_HIGH`, the
`GATE-NOT-DIGIT` agreement state, the `"✗ gate"` CHARACTER DETAIL flag,
and the `[G?]` PER-MODEL SUMMARY marker are all removed from
`ocr_pipeline_mnist.py` — not left in as dead code alongside the new
router path. `mnist_digit_gate.py` itself does not exist as a separate
file in v3; it's superseded by the 5 router scripts
(`v3_mnist_router_ranger_{16,28,32,64,128}.py`). Reasoning:
1. Part 3's own title is "Router/classifier model rewrite," and its
   file-naming section explicitly states `v3_mnist_router_ranger_*.py`
   is "adapted from v2's `mnist_digit_gate.py`" — lineage language for a
   single evolving artifact, not a new sibling built next to the old one.
2. The task frames the router as the gate's scope *expanded* (2-class →
   4-class), and an expansion of an artifact's own scope is that
   artifact changing, not a second artifact appearing beside it.
3. Running both would mean deciding a non-trivial interaction the task
   itself flags as needing resolution ("does the router replace the
   gate's not-digit call, or run after it on boxes the gate calls
   digit/unsure?") for a combination (2-class gate + 4-class router, both
   live) that has no real use case once the router already does
   everything the gate did (digit vs. not-digit) plus more.

Practical effect on `??`: in v3, `??` means only "ensemble split" —
v2's second cause (a confident gate not-digit call) can no longer occur,
because the feature that produced it doesn't exist anymore. This is
documented in `ocr_pipeline_mnist.py`'s own updated "Output markers"
docstring section, alongside the new `[UNK]`, `UC`, and `LC` markers.

Every site that referenced `GATE-NOT-DIGIT`/`gate_verdict`/`gate_prob`
in `ocr_pipeline_mnist.py` was located (via direct read, not grep-and-
assume) and given a router-equivalent, per the task's own checklist:
ensemble voting output (`ROUTER-UNKNOWN`/`ROUTER-UC`/`ROUTER-LC` skip the
ensemble the same way `GATE-NOT-DIGIT` did), CHARACTER DETAIL
(`"✗ router"` for `[UNK]`, new `"○ UC"`/`"○ LC"` flags), PER-MODEL
SUMMARY (`[R?]`/`[UC]`/`[LC]` markers, replacing `[G?]`), and accuracy
scoring (all three router-diverted states excluded from the
correct/total count, each with its own warn line — `[UNK]` gets "N
character(s) flagged unknown by the router," UC/LC get a separate "N
character(s) classified as letters... excluded from digit accuracy
count" line, since being classified as a letter isn't a failure state
the way `[UNK]`/non-digit are).

### Router training data and optimizer

Digit class (label 0): the same merged digit ensemble sources as
`v3_mnist_digit_*.py` at the router's own resolution tier (via
`digit_sources_for_tier()`), individual digit identity discarded.
UC/LC classes (labels 1/2): EMNIST Balanced, restricted to byclass 10-61
(the digit portion of Balanced, byclass 0-9, is intentionally not used —
the router's digit class is sourced entirely from the merged digit
ensemble data instead, for consistency with the digit ensemble and to
avoid diluting it with a third, lower-diversity digit source).

Class balance generalizes the v2 gate's `WeightedRandomSampler` idiom
from 2 to 3 classes — genuinely, not just a copy-paste, since the
gate's two source blocks (digit ConcatDataset, then EMNIST-letters
ConcatDataset) were each internally homogeneous in label, letting it use
one shared weight per block. The router's letters block mixes UC and LC
samples in the shuffled order `random_split()` produced, so
`compute_router_sample_weights()` computes a genuine per-sample weight
for every letter sample based on its actual case label, not a block-
uniform one.

**Optimizer switch: AdaBelief → Ranger (RAdam + Lookahead).** Neither the
sqrt batch-size LR scaling, the `REFERENCE_BATCH_SIZE=256` anchor, nor
the up-to-300-step steps-per-epoch-derived warmup from the gate's
AdaBelief tuning were assumed to carry over — re-derived for Ranger
specifically, per the task's explicit instruction:
- **LR scaling: linear (capped at 4x), not sqrt.** AdaBelief's sqrt
  scaling was chosen because an adaptive per-parameter optimizer
  partially self-compensates for batch size via its own second-moment
  estimate. RAdam is also Adam-family and shares that property, but the
  actual mechanism this restructure leans on is different: RAdam's own
  analytic variance rectification (the `rho_t` term — see the `RAdam`
  class inside each `v3_mnist_router_ranger_*.py` file) is what handles
  the early-training adaptive-LR instability AdaBelief's sqrt scaling
  was partly compensating for, so
  batch-size-to-LR scaling here follows the standard linear-scaling-rule
  literature (Goyal et al., 2017) instead, capped at 4x to avoid
  instability at this project's largest auto-detected batch sizes (into
  the thousands at low resolution).
- **Warmup: a short, fixed 100 steps, not a dynamic up-to-300-step
  derivation.** RAdam's rectification is an *analytic, automatic* warmup
  in its own right — that's its headline property. An external warmup
  schedule on top of it no longer needs to carry the full early-training
  variance-control burden the gate's warmup did for AdaBelief; the fixed,
  short warmup used here exists mainly to give Lookahead's own slow-
  weight mechanism a stable first few steps, which doesn't scale with
  steps-per-epoch the way a primary variance-control warmup would.
- Lookahead `k=6`, `alpha=0.5` — the published Ranger defaults, used
  as-is (no project-specific reason to deviate found or needed).
- `ROUTER_UNSURE_FLOOR = 0.6` (new constant, confidence-threshold
  approach — see below) — chosen to sit meaningfully above the 3-class
  uniform-chance baseline (~0.33) without over-flagging ordinary
  confident predictions; a starting point, not a value tuned against
  real-world data, matching how the gate's own `GATE_UNSURE_LOW`/`HIGH`
  band was originally chosen. Real-world tuning is out of scope for this
  restructure (see "Verification" below) — the user owns that as a
  follow-up, same as for the rest of the ensemble.

**`[UNK]` handling: confidence threshold, not a trained reject class** —
resolved decision, implemented exactly as specified. The router's output
head is 3-class (digit/UC/LC); at inference, a top-class confidence below
`ROUTER_UNSURE_FLOOR` produces `[UNK]` instead of the top class. Chosen
over a trained 4th "reject" class for the same two reasons the task
gives: no negative/reject training data exists anywhere in this project
to train a 4th class on, and it keeps the router's uncertainty handling
consistent with the rest of the project's existing confidence-band/floor
conventions (`NON_DIGIT_CONF_FLOOR`, the v2 gate's own band) rather than
introducing a second, different mechanism.

### Box detection review for letters

`get_boxes()`'s `SIZE_WINDOWS` and aspect-ratio bounds were tuned only
against digit test images in v1/v2. Checked (reasoned through
explicitly, not assumed fine) before treating box detection as
router-ready:
- **Ascender/descender letters** (b, d, h, k, l; g, j, p, q, y): their
  bounding-box height is comparable to or taller than a digit's in the
  same handwriting sample (digits already span roughly cap-height to
  baseline) — no change needed to `SIZE_WINDOWS`' height percentages.
- **Narrow letters** (i, l): a single thin pen stroke has a roughly
  fixed pixel width regardless of a character's design height, so its
  aspect ratio (w/h) can run lower than a typical digit's — the original
  floor (`aspect > 0.1`) risked rejecting a genuinely narrow letter as
  noise. **Changed:** the aspect-ratio floor (now a named constant,
  `MIN_ASPECT_RATIO`) was lowered from 0.1 to 0.06 — low enough to admit
  a narrow single-stroke character, still well above what a pure noise
  speck would realistically produce based on this project's own observed
  digit boxes. `MAX_ASPECT_RATIO` (10.0, guards against wide merged-
  character blobs / scan artifacts) is unrelated to the letter case and
  was left unchanged.
- The touching/merged-character split pass (`_split_oversized_boxes`)
  operates on generic ink-column geometry with no digit-specific
  assumption baked in — checked and left unchanged.
- `MIN_ABS_HEIGHT_PX`/`MIN_ABS_WIDTH_PX` and the width-ceiling rescue
  pass are both about very-small vs. very-wide boxes, orthogonal to the
  narrow-letter case — left unchanged.

This is a documented judgment call based on reasoning about letter
shapes and this project's own existing digit-box statistics, not a
real-world letter-image validation run — see "Verification" below.

### Resume/checkpoint correctness implications

- **Digit ensemble (SOAP, AdamW):** no architecture, optimizer-state
  shape, or scheduler-state shape changed for either kept optimizer —
  old-format v2 checkpoints for SOAP/AdamW are structurally compatible
  with the v3 resume code path (same keys: `epoch`, `optimizer_state`,
  `scheduler_state`/none, `history`, etc.), though there is no reason to
  actually point v3 scripts at v2 checkpoint files, since the task
  requires every v3 model to train from scratch regardless (no weight
  carryover) — this compatibility note is about the *resume format*,
  not an endorsement of reusing old weights.
- **Muon:** brand new optimizer, no prior checkpoint format exists to be
  compatible or incompatible with. `Muon.state_dict()`/`load_state_dict()`
  are inherited unchanged from `torch.optim.Optimizer`'s own ordinal-
  position-based serialization — verified by tracing through
  `Optimizer.__init__`'s handling of `self.state`/`self.param_groups`
  (both populated exactly as any built-in optimizer would populate them),
  not by an actual save/load round-trip (no GPU/PyTorch available in
  this environment — see "Verification" below).
- **Router (Ranger):** same situation as Muon — brand new, no prior
  format. `Lookahead.state_dict()` delegates entirely to the wrapped
  `RAdam` instance's own `state_dict()`; the Lookahead-added
  `slow_buffer` per-parameter state and `la_step_counter` per-group
  state were deliberately stored INSIDE the base optimizer's own
  `self.state`/`param_groups` (via `self.state = base_optimizer.state`)
  specifically so they ride along with `RAdam`'s existing ordinal-
  position serialization instead of needing separate plumbing — traced
  through by hand, not verified via an actual resume round-trip.
- **Batch-size-mismatch-on-resume check** (Muon, SOAP already had this
  implicitly via full state resume; router explicit): a resumed run
  whose auto-detected batch size differs from the one that saved the
  resume file skips restoring optimizer/scheduler state (which encodes a
  batch-size-dependent LR/schedule) but still resumes model weights,
  epoch count, and patience state — same precedent as the v2 gate's own
  check, now also present in the Muon and router scripts (SOAP's own
  resume block does not carry this check in v2 or v3 — flagged here as
  an existing SOAP-specific gap, not introduced by this restructure).

### Verification — what was and wasn't actually run

**No GPU, no PyTorch, no dataset, and no training run was available in
the environment this restructure was built in** — this is a plain
container with no `torch`/`numpy`/`onnx` installed and no access to the
project's own hardware or the `E:\CSC-114\...`-rooted datasets. Nothing
below should be read as "this was trained and confirmed working, only
the code was written and hoped to work." What actually was checked:

- **`python3 -m py_compile`** on all 31 v3 `.py` files (`common/`'s 9
  modules, `supplementary_data.py`, `ocr_pipeline_mnist.py`, and all 20
  training scripts — 15 digit + 5 router) — confirms syntax validity and
  that every file is at least importable in principle, not that any
  function actually runs correctly end-to-end. `pyflakes` re-run after
  every correction pass in this changelog to catch unused-import/
  undefined-name regressions — found and fixed dead `RAM_RESERVE_GB`/
  `os`/`math`/`onnx` imports reintroduced by regenerating from a fresh
  template this way. When `RAdam` (which calls `math.sqrt`) was inlined
  directly into each `v3_mnist_router_ranger_*.py` (see "Naming and
  modularization-scope correction" below), the resulting missing
  `import math` was actually caught by manual inspection first (checked
  proactively, knowing `RAdam` needed it, before `pyflakes` was run
  against those specific files) — confirmed separately afterward that
  `pyflakes` does flag a genuinely undefined name used only inside a
  method body (`py_compile` alone does not, since it doesn't execute
  function bodies), so it would have caught this one too if the manual
  check had missed it.
- **`ast.parse()`** used separately to enumerate `ocr_pipeline_mnist.py`'s
  top-level functions/classes and confirm the expected surface (e.g.
  that `predict_gate`/`GATE_UNSURE_LOW` etc. are actually gone, not just
  renamed inconsistently) — a real static check, not a guess.
- **`BALANCED_TO_BYCLASS`'s new lowercase mapping** was verified
  numerically with a standalone, torch-free Python snippet reproducing
  just that dict's construction logic and printing every byclass 36-61
  entry alongside its letter, checked by hand against the known
  a/b/d/e/f/g/h/n/q/r/t list (11 entries, matching EMNIST Balanced's
  documented class count).
- **The Muon architecture's parameter count** (~9.68M, quoted in its own
  module docstring and the standard-settings table below) was computed
  by hand from each layer's `Conv2d`/`BatchNorm2d`/`Linear` shape
  formulas in a standalone script — not read off an actual instantiated
  `nn.Module`, since no `torch` is available here to instantiate one.
- **Cross-file label-ordering consistency** (`ROUTER_LABELS` in
  `ocr_pipeline_mnist.py` vs. `LABEL_MAP` in
  `v3_mnist_router_ranger_*.py`, both `["digit", "UC", "LC"]`) was
  checked by direct comparison of both literals, since this is a
  contract enforced by convention, not by any shared import, and a
  mismatch would silently mislabel every letter box.

**What was NOT verified and still needs confirming on real hardware:**
an actual training run for any of the 16 v3 scripts (auto batch-size
detection, the full data-loading path including the real
`E:\CSC-114\...` datasets, an actual resume-from-checkpoint round-trip,
real GradScaler/AMP behavior with the new `Muon`/`Lookahead` optimizer
classes, and real ONNX export/inference for all 13 models); Muon's and
Ranger's actual convergence behavior and final accuracy — the
hyperparameters recorded in the standard-settings table below are
reasoned starting points, not results; the router's real precision on
actual scanned letters, and whether `ROUTER_UNSURE_FLOOR=0.6` and the
lowered `MIN_ASPECT_RATIO=0.06` are actually well-tuned against real
images rather than just plausible on paper. Real-world validation is
explicitly out of scope for this restructure task (per the task's own
constraints) — training and validating against real data is a follow-up
step the user owns.

### Naming consistency

Every v3 file uses the `v3_mnist_digit_{optimizer}_{resolution}` /
`v3_mnist_router_ranger_{resolution}` convention from the start — there
is no historical naming to preserve within v3 itself (unlike v2's own
`ocr_soap_*` → `mnist_soap_*` rename, which is a v2-only historical note,
left as-is in v2's own changelog and not revisited here). See "File
change list" below for the full v3-file → v2-source-file adaptation
mapping, so lineage stays traceable even though the v2 files themselves
were never touched (this restructure worked from a separate local output
location per the task's own instructions — the v2 repo is untouched).

`ocr_pipeline_mnist.py`'s `short_model_name()` regex now tries the new
`v3_mnist_digit_{optimizer}_{res}` and `v3_mnist_router_{optimizer}_{res}`
patterns first, then falls back to the old v2 `v1_{optimizer}_{res}_{res}`
and bare `{optimizer}_{res}` patterns (e.g. `adahessian_64.onnx`,
`soap_128.onnx`) — kept for robustness (the pipeline can still be pointed
at old-format `.onnx` files without crashing its name-parsing), not
because any v2-named file is expected to actually be used going forward.

### Dependency verification tooling added (post-delivery, per direct user follow-up)

**Gap:** the initial v3 delivery had no standalone dependency file —
only `README.md`'s "Requirements" prose section, which lists packages
but isn't installable or scriptably checkable. User asked directly where
the dependency file was so they could verify their own environment.

**Added, both per explicit confirmation (not guessed at):**
- `requirements.txt` — plain `pip install -r requirements.txt` list,
  built from the same verified-against-actual-imports set already in
  `README.md`'s Requirements section (re-confirmed via a fresh grep of
  every top-level `import`/`from` line across all 31 `.py` files before
  writing this, not copied from the README without re-checking). Notes
  up front that `torch`/`torchvision` need the CUDA-specific wheel index
  first, since a bare `pip install -r requirements.txt` would otherwise
  silently pull a CPU-only torch build from PyPI.
- `setup_packages.py` — a v3-native port of v2's same-named script (see
  "File change list" — this was flagged as a known gap in the original
  delivery and deliberately not built then, since it wasn't part of the
  restructure's required output). Same `--check`/`--skip-torch`/`--cuda`/
  `--skip-onnx` flag shape as v2's version, but with v3's actual package
  list: `adabelief-pytorch` and `lion-pytorch` dropped (AdaBelief/Lion
  aren't in the v3 roster), no entry added for Muon or Ranger (both are
  inlined in their training scripts, no PyPI package involved for
  either), and the Windows-only `wmi`/LibreHardwareMonitor optional
  group removed entirely — v3's `common/telemetry.py` only ever reads
  `nvidia-smi` + `psutil`, unlike v2's `benchmark_cudnn_speed.py` (not
  carried into v3), which was the only v2 file that used WMI CPU-temp
  telemetry. `onnx`'s check/install reasoning was narrowed to accurately
  say it's needed by the 5 SOAP digit scripts' `validate=True` export
  path specifically (confirmed via grep — only the SOAP scripts pass
  `validate=True` to `common/onnx_export.py`'s `export_onnx()`), not "for
  training" generally.

**Verification:** both files checked against a fresh `grep` of every
top-level import across `common/*.py` and every top-level `.py` file,
not assumed from the README. `setup_packages.py` compiled clean via
`python -m py_compile` and `python -m pyflakes`.

### Minimum steps/epoch floor added to all 20 training scripts (post-delivery, per direct user follow-up)

**Reported, with a real run's console output:** a real run of
`v3_mnist_router_ranger_16.py` (RTX 4080) auto-detected batch size 16384
— the top of the candidate ladder — and trained at only **5 batches per
epoch** (81,678 combined router-training samples ÷ 16384), which is too
few real gradient-update steps per epoch. User asked for a hard floor of
at least 15 steps/epoch.

**Root cause:** `determine_batch_size()` (`common/batch_sizing.py`)
probes VRAM/RAM headroom using dummy random tensors, *before* the real
dataset is loaded (every script determines batch size first, then loads
its dataset — see each script's `run_training()`) — it has no way to
know the real training-set size at the point it picks a batch size. A
small model (the router, ~62.6K params) or a small dataset (16×16's
USPS-only tier) can land on a batch size that's fine for VRAM but far
too large relative to the actual number of training samples, without
this ever being caught anywhere in the pipeline.

**Fix:** added `cap_batch_size_for_min_steps(batch_size, dataset_size,
min_steps)` to `common/batch_sizing.py` — a pure post-hoc cap
(`min(batch_size, dataset_size // min_steps)`, never raises, only
lowers). Applied identically in all 20 training scripts (15 digit + 5
router), each with its own local `MIN_STEPS_PER_EPOCH = 15` constant
(matching the project's existing convention of per-script hyperparameter
constants like `PATIENCE`, not a shared value baked into `common/`) —
inserted right after the real training dataset is loaded (`train_ds`)
and before the train `DataLoader` is built, so `cfg["batch_size"]` (and
therefore the LR-scaling formulas and dataloaders that read it
downstream) reflects the capped value consistently for the rest of the
run. Prints `[Batch] Capping batch size X -> Y ...` when the cap
actually lowers anything, so the transcript records when and why.

**Verified against the real reported case:** router at 16×16, 81,678
train samples, `MIN_STEPS_PER_EPOCH=15` → caps 16384 down to 5,445 →
81,678 / 5,445 ≈ 15.0 steps/epoch, confirmed by hand calculation before
shipping.

**Bug caught and fixed before shipping, not after:** the first pass at
this edit (via a scripted regex insertion across all 20 files) produced
an f-string with doubled curly braces (`f"...{{cfg['batch_size']}}..."`)
— a copy-paste artifact from writing the block as a plain string
template rather than directly as target-file Python. Doubled braces in
an f-string are Python's own escape for a *literal* brace character, so
this would have printed the literal text `{cfg['batch_size']}` instead
of the real number in every script's console/transcript output. Caught
by rereading the actual patched output (not assumed correct because the
regex substitution count matched), fixed with a second scripted pass
across all 20 files, and reverified via `grep` that zero double-brace
occurrences remain.

**Verification:** all 20 files (plus `common/batch_sizing.py`)
recompiled clean via `python -m py_compile`. Confirmed via `grep` that
`MIN_STEPS_PER_EPOCH = 15` and `cap_batch_size_for_min_steps` both
appear in exactly the expected files, and that the fixed f-string block
reads correctly (single braces) in every file, not just the one spot-
checked directly.

### DataLoader worker count auto-scaled to real core count (post-delivery, per direct user follow-up)

**Requested directly:** auto-adjust `NUM_WORKERS` based on this
machine's actual core/thread count, leaving 25% free, rather than the
hardcoded `10` every script had — on a 24-thread CPU that left 14 threads
completely idle beyond the existing 25% OS reservation; on a smaller CPU
it could over-subscribe past the reservation entirely.

**Fix:** added `usable_cpu_count(cpu_reserve_pct=25)` to
`common/seeding.py` — the same reservation math `reserve_cpu_threads()`
already used for `torch.set_num_threads()` (`max(1, total_cores -
round(total_cores * cpu_reserve_pct / 100))`), factored out and exposed
standalone, deliberately NOT gated on CUDA availability the way
`reserve_cpu_threads()` is (DataLoader worker processes compete for CPU
regardless of whether model compute happens on GPU or CPU).
`reserve_cpu_threads()` itself now calls this helper internally — no
behavior change there.

Wired into all 20 training scripts, which split into two genuinely
different existing patterns (per the same adamw/router vs. soap/muon
sub-family structural split documented elsewhere in this changelog):
- `v3_mnist_digit_adamw_*.py` / `v3_mnist_router_ranger_*.py` (10 files):
  had a `NUM_WORKERS = 10` module constant — changed to
  `NUM_WORKERS = usable_cpu_count()`.
- `v3_mnist_digit_soap_*.py` / `v3_mnist_digit_muon_*.py` (10 files): had
  no named constant, `num_workers=10` was hardcoded directly inline in
  the train `DataLoader(...)` call — added a `NUM_WORKERS =
  usable_cpu_count()` constant (matching the other 10 files' naming) and
  updated the inline call to `num_workers=NUM_WORKERS,
  persistent_workers=NUM_WORKERS > 0`.

Val/test loaders are deliberately unaffected — they already correctly
use `num_workers=0` everywhere (unparallelized, small enough not to need
it), which was true before this change and stays true after.

**Verification:** all 20 files recompiled clean via `python -m
py_compile`, pyflakes-clean. Confirmed via `grep` that `usable_cpu_count`
appears at least twice (import + usage) in every one of the 20 files.

### Hardware telemetry expanded: power/clocks/throttle, memory/disk, min-avg-max sampling, data-wait timing (post-delivery, per direct user follow-up)

**Requested directly, with all four proposed options selected:** more
granular hardware telemetry per epoch. The original `get_hw_stats()`
(GPU utilization/temp via `nvidia-smi`, VRAM peaks via `torch.cuda`,
CPU/RAM via `psutil`) sampled exactly ONE snapshot per epoch, at the
midpoint batch — a real run's own transcript (router, 16×16) showed this
landing on an idle spot more than once (`CUDA 0%` at one epoch, `5%` at
another), meaning the logged number could misrepresent the epoch's real
utilization rather than just being coarse.

**Added to `common/telemetry.py` — `HardwareMonitor` class** (kept the
original `get_hw_stats()` untouched alongside it, for any caller that
still just wants one cheap snapshot):
- `.sample()` — one `nvidia-smi` call per invocation gathering
  `utilization.gpu`, `utilization.memory` (bandwidth pressure, distinct
  from compute utilization — the original only had compute),
  `temperature.gpu`, `power.draw`, `clocks.sm`, `clocks.mem`,
  `fan.speed`, and three `clocks_throttle_reasons.*` flags (hw/sw
  thermal slowdown, hw power-brake) collapsed into one `gpu_throttled`
  flag — all in a single subprocess call, not one call per new field,
  since `nvidia-smi` is already called multiple times per epoch now (see
  below) and multiplying subprocess overhead by field count would have
  been a real cost. Plus disk I/O as a rate (MB/s), computed from the
  delta against the instance's own previous sample divided by elapsed
  time (`psutil.disk_io_counters()` itself only returns cumulative bytes
  since boot, not a per-interval rate).
- `.epoch_summary(samples)` — reduces a list of `.sample()` calls into
  min/avg/max for the genuinely volatile metrics (utilization, temp,
  power, CPU%, RAM used, disk throughput), avg-only for the less-volatile
  clock speeds and fan speed, max for the throttle flag (1 if throttled
  at ANY sampled point in the epoch, not just the last), and last-value
  for things that don't change within a run (`ram_total_gb`) or are
  already cumulative peaks (`vram_peak_alloc_gb`/`vram_peak_reserved_gb`,
  unchanged semantics from the original `get_hw_stats()`). Ignores -1
  (unknown) readings when reducing, so a field this particular card
  doesn't expose reports -1 for min/avg/max rather than being dragged
  down by treating -1 as a real zero.

**Sampling frequency, in every `train_one_epoch()`:** replaced the
single midpoint-only check with 5 sample points spread across the epoch
(10/30/50/70/90% of `num_batches`), collected into `_hw_samples` and
reduced via `epoch_summary()` at the end of the epoch — directly
addresses the idle-spot-misrepresenting-the-epoch problem the real
router transcript surfaced.

**Data-wait vs. compute time, also in every `train_one_epoch()`:** times
the dataloader's next-batch yield separately from the actual training
step (`_data_wait_s` / `_compute_s`, accumulated across the epoch) —
this is a genuinely different diagnostic than utilization%: a GPU
sitting at low utilization because it's *starved waiting on data*
(CPU/disk-bound) looks identical in utilization% terms to one that's
simply not doing much work, but the wait/compute split tells them apart.
Deliberately NOT added to `common/telemetry.py` itself — only the caller
knows where its own "waiting for the next batch" boundary actually is,
which differs between SOAP's plain first-order backward and every other
optimizer's `amp_train_step()` call.

**CSV schema (`common/cli_logging.py`):** `save_log()`'s hardcoded
7-field telemetry list replaced with a new `HW_TELEMETRY_FIELDS`
constant (33 columns: the full `epoch_summary()` output plus
`data_wait_s`/`compute_s`) — `save_log()` now iterates this list
generically instead of listing every field by name twice (once as a
dict key, once in the fieldnames list), so a future telemetry field only
needs to be added in one place.

**`run_training()`'s history-recording block, all 20 scripts:** the old
explicit 6-field `history.setdefault(...)` list — which would have
silently dropped every new telemetry field without a matching edit here
— replaced with `for hw_key, hw_val in hw.items(): history.setdefault(hw_key,
[]).append(hw_val)`, so it automatically stays in sync with whatever
`HardwareMonitor.epoch_summary()` produces.

**Per-epoch console print line, all 20 scripts:** rewritten to surface
the new headline numbers (CUDA avg/max%, temp avg/max, avg power, avg
RAM, data-wait/compute split, a `[THROTTLED]` tag when
`gpu_throttled_any == 1`) without dumping all 33 CSV columns into every
console line — full detail is in the CSV, the console line stays
scannable.

**Bug caught and fixed before shipping, not after:** the first version
of `epoch_summary()`'s min/avg/max reduction omitted `ram_used_gb`
entirely — it was in the original `get_hw_stats()` output and was meant
to carry forward, but got left out of the reduction field list while
`cuda_util_pct`/`gpu_temp_c`/etc. were added around it. Caught by
diffing the actual `epoch_summary()` output against `HW_TELEMETRY_FIELDS`
in a standalone test (not assumed correct because both lists were
written by the same pass) before wiring this into any of the 20 scripts,
fixed by adding it to the reduction field list, and reverified via the
same diff afterward.

**NOTE ON VALIDATION (stated directly in `common/telemetry.py`'s own
docstring too, not just here):** `utilization.gpu`/`temperature.gpu`
have been confirmed working against this project's real hardware (RTX
4080) in prior transcripts. The new fields — `power.draw`,
`clocks.sm`/`clocks.mem`, `fan.speed`, `clocks_throttle_reasons.*`,
`utilization.memory` — are standard, documented `nvidia-smi
--query-gpu` fields, but have NOT yet been confirmed against a real run
in this project (no GPU available in the environment this was built in).
Parsing is defensive — each field independently falls back to -1 if
missing, `"N/A"`, or the whole `nvidia-smi` call fails — so an
unsupported field on a given card degrades to -1 rather than crashing
the run, but the actual values these produce should be checked against
a real run before being trusted in a report.

**Verification:** `HardwareMonitor.epoch_summary()`'s reduction logic
unit-tested standalone (mocked `torch`/sample dicts, since no GPU is
available in this environment) against both a realistic mixed-value
sample set and an all-unknown (-1) sample set, confirming correct
min/avg/max exclusion of -1 readings and correct degradation when
nothing is available. The full per-epoch print f-string dry-run tested
standalone against both a realistic and an all-(-1) mock `hw` dict to
confirm no `KeyError`/format crash either way before touching any of the
20 real files. Verified byte-identical anchor text across all 20 files
for every substitution point (train_one_epoch's setup block, sampling
point, tail, and run_training's print/history block) before applying
any scripted edit — one real difference was found (digit scripts vs.
router scripts have a blank-line formatting difference in
train_one_epoch's tail) and both variants were handled explicitly rather
than forcing one pattern. All 20 training scripts plus
`common/telemetry.py` and `common/cli_logging.py` recompiled clean via
`python -m py_compile` and pyflakes-clean (aside from a pre-existing,
already-`# noqa`-documented unused import in `ocr_pipeline_mnist.py`,
unrelated to this change). Grepped all 20 files afterward for any
leftover reference to the removed `mid_epoch_hw`/`midpoint_idx`/
`get_hw_stats(` names — zero remaining.

**Files changed:** `common/telemetry.py` (new `HardwareMonitor` class,
`get_hw_stats()` kept unchanged), `common/cli_logging.py` (`save_log()`
CSV schema), `common/seeding.py` (this entry's `usable_cpu_count()`
addition, see above), all 20 training scripts (`train_one_epoch()`
sampling/timing, `run_training()` print line and history recording,
import line).

### CPU core count / worker allocation was never printed on a GPU run (post-delivery, per direct user follow-up)

**Reported directly:** asked where in the console output core count and
DataLoader worker allocation are shown. Checked rather than assumed —
answer was nowhere, on a GPU run. `setup_device()` (`common/
telemetry.py`) only printed CPU core/reservation detail inside its
`else:` (CPU-only) branch; the GPU branch printed only GPU name/VRAM/
CUDA version/AMP status. Since `NUM_WORKERS` (every script, all 20,
via the `usable_cpu_count()` addition earlier in this changelog) is
computed and used as DataLoader worker count regardless of which device
model compute runs on, a GPU run — the normal case on this project's own
RTX 4080 — never told you how many cores were detected or how many
workers were actually allocated.

**Fix:** moved the core-count/reservation print out of the `else:`
branch to run unconditionally, before the GPU/CPU branch, and reworded
it to explicitly name the connection to `NUM_WORKERS`:
`[CPU] {total} logical cores detected — reserving {reserved} ({pct}%)
for the OS, {usable} usable (this machine's NUM_WORKERS = {usable}
DataLoader workers; also the thread count torch.set_num_threads() uses
on CPU-only runs)`. The GPU branch's existing `[Device]` line and the
CPU branch's line are both unchanged apart from the CPU branch's line
being shortened (no longer needs to repeat the reservation math the new
unconditional `[CPU]` line above it already states).

Single fix in `common/telemetry.py` — every one of the 20 training
scripts calls `setup_device()` from this shared module with its default
`cpu_reserve_pct=25` (confirmed via grep — no script passes a different
value), so no per-script changes were needed; all 20 pick this up
automatically.

**Verification:** dry-run of the new print line's exact formatting
against this project's real hardware numbers (24 logical cores, 25%
reserved) confirms `[CPU] 24 logical cores detected — reserving 6 (25%)
for the OS, 18 usable (this machine's NUM_WORKERS = 18 DataLoader
workers...)` — matches the real `NUM_WORKERS` value every script's
`usable_cpu_count()` call already computes, since both use the same
reservation formula. `common/telemetry.py` recompiled clean via
`python -m py_compile`, pyflakes-clean.

**Files changed:** `common/telemetry.py` only.

---

### Multi-GPU support added: parallel independent runs (--gpu) + real DDP (post-delivery, per direct user follow-up)

**Requested directly, with both options selected after being presented as
genuinely different scopes:** (A) run different models on different GPUs
at once, and (B) split ONE model's training across multiple GPUs
(`DistributedDataParallel`).

**⚠️ Part B (DDP) is NOT YET VALIDATED against real multi-GPU hardware —
stated here, in `common/distributed.py`'s own module docstring, and in
`README.md`.** Every real number anywhere in this changelog — every
batch size, every VRAM figure, every accuracy, every telemetry reading —
came from a single RTX 4080. No multi-GPU machine was available to
confirm any of Part B against. Part A (`--gpu`, single-process,
independent runs) is much lower-risk — it's the same code path every
existing single-GPU run already exercises, just pointed at a different
physical card — and is correspondingly more trustworthy, though still
unconfirmed on a real 2-GPU box.

#### Part A: `--gpu N` for parallel independent runs

Every one of the 20 scripts gained a `--gpu` flag
(`python v3_mnist_digit_soap_64.py --gpu 1`). `setup_device()` (`common/
telemetry.py`) now calls `torch.cuda.set_device(gpu_id)` before anything
else, prints which physical GPU was selected (`[Device] GPU 1/1: ...`),
and warns if multiple GPUs are detected but none was explicitly chosen.

**Real, pre-existing bug found and fixed while building this, not
introduced by it:** every `nvidia-smi --query-gpu=...` call in `common/
telemetry.py` and `common/batch_sizing.py`, and every
`torch.cuda.get_device_properties(0)` call, hardcoded either no `-i`
index (querying and returning a CSV row for EVERY GPU on the system) or
literal index `0` (always the physical first GPU). On this project's own
single-GPU dev machine this happened to parse and behave correctly by
accident — one GPU means one CSV row and index 0 is the only real index
— but on ANY machine with 2+ GPUs, the nvidia-smi calls would have
returned multiple rows where the code expected one (silently mis-parsing
the wrong values), and `get_device_properties(0)` would have reported
GPU 0's specs even when training was actually running on GPU 1. Fixed by
adding a `_cuda_index()` helper (`common/telemetry.py`, wraps
`torch.cuda.current_device()`) used everywhere telemetry queries the
GPU, and by threading the real `device.index` through `common/
batch_sizing.py`'s probe instead of a literal `0`.

#### Part B: real DDP, launched via `torchrun`

New `common/distributed.py` — rank/world-size env-var helpers (matching
`torchrun`'s `RANK`/`WORLD_SIZE`/`LOCAL_RANK` contract), process-group
init/teardown, `wrap_model_ddp()`/`unwrap_model()`, `all_reduce_sum()`
for aggregating per-rank metrics, `broadcast_int()` for the batch-size
probe, and `DistributedWeightedRandomSampler` — a standard pattern (not
a novel invention) combining every training script's existing
class-balancing `WeightedRandomSampler` with DDP's per-rank sharding
requirement: every rank deterministically draws the same full weighted
sample (seeded by `seed + epoch`, no inter-process communication needed
for this part) then takes `indices[rank::world_size]`, guaranteeing
disjoint per-rank work with the original per-sample weights preserved in
aggregate. Verified via pure-Python simulation (disjoint coverage, no
duplicates, `__len__` matches real slice length including the
not-evenly-divisible case) since no second GPU/process was available to
verify real multi-rank behavior against.

**Integration, all 20 scripts:**
- `setup_device()`'s `gpu_id` comes from `get_local_rank()` instead of
  `--gpu` when running distributed (the launcher already assigns each
  process its device — don't also pass `--gpu` under `torchrun`).
- `setup_distributed(device)` called right after device resolution.
- Model wrapped via `wrap_model_ddp()` right after construction.
- Train-loader's `WeightedRandomSampler` swapped for
  `DistributedWeightedRandomSampler` when distributed; `sampler.
  set_epoch(epoch)` called every epoch (the standard DDP sampler
  contract — without it, every epoch draws the identical weighted
  sample).
- `train_one_epoch()`'s `total_loss`/`total_correct`/`total_samples`
  all-reduced (summed across ranks) before computing the final loss/acc
  ratios — each rank only sees its own shard, so without this a
  distributed run's reported per-epoch numbers would silently reflect
  one rank's slice, not the real global epoch.
- `evaluate()` deliberately left UNCHANGED — val/test loaders are NOT
  sharded (every rank evaluates the full val/test set independently),
  so no all-reduce is needed there; less efficient than sharding eval
  too, but trivially correct without any cross-rank synchronization risk.
- `determine_batch_size()` (`common/batch_sizing.py`) now probes ONLY on
  rank 0 under DDP and broadcasts the result to every other rank via
  `broadcast_int()`, rather than trusting every rank's independent probe
  to land on the same number (only guaranteed if every GPU in the job is
  identical — unverifiable from inside this codebase) — also cuts
  probe-time VRAM churn on every rank but one.
- `cap_batch_size_for_min_steps()` gained a `world_size` parameter — the
  real steps/epoch under DDP is `dataset_size / (per_rank_batch *
  world_size)`, since every rank steps together in lockstep.
- Router's `scaled_learning_rate()` call site now scales from the
  GLOBAL effective batch size (`cfg["batch_size"] * get_world_size()`),
  the standard linear-scaling-rule convention for distributed training —
  DDP's backward-pass gradient averaging is already computed over the
  global batch, so the LR should reflect that, not just one rank's slice.
  The resume-file batch-size compatibility check (see
  `common/checkpointing.py`'s own docstring on this precedent) was
  updated to compare against this same global figure, not the per-rank
  one — a resume where `world_size` changed between runs but the
  per-rank batch size happened to auto-detect identically would
  otherwise have passed the check while actually resuming optimizer/
  scheduler state built for a different real LR.
- All file I/O — `EarlyStopping`'s checkpoint save, `save_resume_state`/
  `clear_resume_state`, `_Tee`'s transcript file, `save_log`,
  `plot_history`, `export_onnx` (all in `common/`, so every one of the
  20 scripts picks this up automatically with no call-site changes
  needed) — gated to rank 0 only. Several ranks all writing the same
  path at once would race; console `print()` output is NOT suppressed on
  non-rank-0 (useful for spotting a hung/crashed rank).
- `cleanup_distributed()` called at the end of every `run_training()`.

**Real bugs found and fixed during this integration, caught by a
deliberate line-by-line re-read of the fully-wired result before
shipping, not assumed correct because every scripted edit's match count
was clean:**
1. **A genuinely broken f-string** (doubled curly braces, from an
   earlier unrelated edit this session) would have printed literal
   `{cfg['batch_size']}` text instead of the real number — caught and
   fixed before this entry (see the min-steps-floor entry above), not a
   DDP-specific bug, mentioned here only because the same
   re-read-before-shipping discipline caught the next three, which ARE
   DDP-specific.
2. **`model.load_state_dict(...)` called directly on the DDP-wrapped
   `model`** at every resume-load and best-checkpoint-reload site (2 per
   digit script, 2 of the router's 3 — the third, inside `--infer`'s
   `classify_image()`, loads into its own always-unwrapped model and was
   never broken). A DDP-wrapped model's `state_dict()` keys are prefixed
   `module.`; the checkpoint files were already being saved unwrapped
   (correctly), so loading them back INTO the wrapped model directly
   would have raised a key-mismatch error the first time any of this
   ran under real `torchrun`. Fixed by routing every one of these 9
   call sites through `unwrap_model(model).load_state_dict(...)` —
   applied uniformly, including the one inside `classify_image()`,
   since `unwrap_model()` is a provably safe identity no-op on a model
   that was never DDP-wrapped in the first place.
3. **The final-model `.pt` save used raw `model.state_dict()`** (same
   `module.`-prefix problem as #2, in the opposite direction — this
   file gets read back into `model_cpu`, a fresh NEVER-wrapped model,  a
   few lines later for ONNX export) **and wasn't rank-gated at all** —
   every rank would have written the same path simultaneously. Fixed:
   `unwrap_model(model).state_dict()`, wrapped in `if is_main_process():`.
   The `unwrap_model` import itself had also been missed entirely when
   the import list was first built — caught by the same read-through,
   confirmed via a project-wide grep showing zero prior occurrences.
4. **The ONNX-export reload block (`model_cpu.load_state_dict(torch.load(
   cfg["final_model_path"], ...))`) ran on every rank**, but — after
   fixing #3 — only rank 0 actually writes that file: every other rank
   would have hit a missing-file error, or raced rank 0's write on a
   from-scratch run. Fixed by wrapping the entire `try/except` block in
   `if is_main_process():` too, not just the save that precedes it.

**Known, accepted limitation, documented rather than engineered around:**
Schedule-Free AdamW's `_batchnorm_warmup()` reads from the now-sharded
`train_loader` under DDP, so each rank warms up its BatchNorm running
stats on only its own data slice, not the full batch — a statistical
imprecision (each rank's BN stats can differ slightly), not a crash or a
silently-wrong metric. Full cross-rank BatchNorm-stat synchronization
(`torch.nn.SyncBatchNorm`) is real, separate PyTorch functionality this
integration does not add — out of scope for what a multi-GPU follow-up
request reasonably calls for; noted here for whoever next touches this,
same as this project's other documented-not-fixed scope boundaries.

**Verification:** `common/distributed.py`'s pure-Python logic (env-var
rank/world-size detection, `unwrap_model()`, the non-distributed
passthrough behavior of `all_reduce_sum()`/`broadcast_int()`, and the
sampler's disjoint-sharding slice math) unit-tested standalone with a
mocked `torch` module, since no real `torch` or GPU is available in this
environment. Every scripted multi-file edit in this whole integration
used the same anchor-verification-before-transform discipline as every
other change in this changelog (exact match counts checked per file
before writing, mismatches reported rather than silently skipped or
force-applied). All 20 training scripts plus `common/distributed.py`,
`common/telemetry.py`, `common/batch_sizing.py`, `common/checkpointing.py`,
`common/cli_logging.py`, and `common/onnx_export.py` recompiled clean via
`python -m py_compile` and pyflakes-clean after every stage. **Real
multi-GPU behavior (process-group init actually succeeding, NCCL
actually working, gradients actually syncing correctly, the sampler's
disjoint-sharding holding up across real ranks) remains unconfirmed —
this needs to be run on real 2+ GPU hardware before being trusted for
an actual multi-day training run.**

**Files changed:** `common/distributed.py` (new), `common/telemetry.py`
(`_cuda_index()`, GPU-index-aware nvidia-smi calls, `setup_device()`'s
`gpu_id` parameter), `common/batch_sizing.py` (GPU-index-aware probe,
rank-0-only + broadcast batch-size determination, `world_size`-aware
min-steps cap), `common/checkpointing.py`/`common/cli_logging.py`/
`common/onnx_export.py` (rank-0 gating), all 20 training scripts
(`--gpu` flag, DDP wiring throughout `run_training()`/`train_one_epoch()`,
router's global-batch LR scaling and resume check). `README.md` updated
with both usage modes.

---

## Standard settings record — all 15 digit models + the router

Recorded so "what were model X's actual settings" can be answered
without re-reading each training script. Every value below is what each
script is CONFIGURED to use — since no training run was possible in this
environment (see "Verification" above), auto-detected batch sizes are
not filled in with real numbers; they will be printed by each script's
own `[Batch] Refined to batch size N` log line on first real run, and
should be recorded here (or in each run's own CSV/CLI-transcript log)
once that happens.

### Digit ensemble — SOAP (kept from v2, unchanged hyperparameters)

Applies identically to `v3_mnist_digit_soap_{16,28,32,64,128}.py`:

| Setting | Value |
|---|---|
| Optimizer | SOAP (Shampoo + Adam, Kronecker-factored second-order), `pytorch_optimizer.SOAP` |
| LR | 1e-3 (fixed, no batch-size scaling — unchanged from v2) |
| Betas | (0.95, 0.95) |
| Weight decay | 5e-4 |
| Precondition frequency | 100 |
| Batch size | auto-detected per run (see `[Batch]` log lines) — not fixed |
| LR schedule | linear warmup 500 steps → cosine decay, `common/scheduler.py`; `total_steps = 200 × steps_per_epoch` (shape target only, not a cap) |
| Mixed precision | **disabled** — Kronecker eigendecomposition requires float32 |
| Gradient clipping | none |
| Patience | 20 |
| Architecture | `OCRConvNetTriplePyramid`, 96→192→384→768, ~4.6M params |
| Label smoothing | 0.05 |

Per-resolution digit sources (via `digit_sources_for_tier()`):
- **16×16:** USPS only, via `load_base_usps()` (USPS's own native
  resolution, the entire spine at this tier — no base MNIST at all).
- **28×28:** MNIST + EMNIST Digits + SVHN + ARDIS IV, via `load_base_mnist()`
  as the spine. **USPS absent.**
- **32×32 / 64×64 / 128×128:** MNIST (spine) + EMNIST Digits + SVHN +
  ARDIS IV + USPS (all five sources).

### Digit ensemble — AdamW (kept from v2, unchanged hyperparameters)

Applies identically to `v3_mnist_digit_adamw_{16,28,32,64,128}.py`:

| Setting | Value |
|---|---|
| Optimizer | Schedule-Free AdamW, `schedulefree.AdamWScheduleFree` |
| LR | 1e-3 (Schedule-Free manages effective LR internally via iterate averaging — no external scheduler) |
| Weight decay | 1e-4 |
| Batch size | auto-detected per run — not fixed |
| Warmup | `warmup_steps = len(train_loader)` (1 epoch), internal to Schedule-Free |
| Mixed precision | enabled (AMP + GradScaler) |
| Gradient clipping | max_norm=1.0 |
| Patience | 15 |
| Architecture | `OCRConvNetWide`, 32→128→256→512, SE attention + StochasticDepth, ~9.7M params |
| Label smoothing | 0.05 |
| BatchNorm handling | 50-batch warm-up pass (`model.train()` + `optimizer.eval()`) before every validation/test evaluation, per Schedule-Free's own requirement |

Per-resolution digit sources: same as SOAP above (16×16 = USPS only via
`load_base_usps()`; 28×28 excludes USPS; 32/64/128 include it).

### Digit ensemble — Muon (new for v3)

Applies identically to `v3_mnist_digit_muon_{16,28,32,64,128}.py`:

| Setting | Value |
|---|---|
| Optimizer | Muon (`Muon` class, inlined in each `v3_mnist_digit_muon_*.py`) — Newton-Schulz orthogonalized momentum for `ndim>=2` params, AdamW fallback for the rest |
| Muon group | lr=0.02, momentum=0.95, nesterov=True, ns_steps=5, weight_decay=0.01 |
| AdamW-fallback group | lr=3e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01 |
| Which params go where | `ndim >= 2` (Conv2d/Linear `.weight`) → Muon; `ndim < 2` (all biases, BatchNorm weight+bias) → AdamW fallback |
| Batch size | auto-detected per run — not fixed |
| LR schedule | linear warmup 300 steps → cosine decay, `common/scheduler.py`, applied to BOTH param groups (each keeps its own base LR ratio); `total_steps = 150 × steps_per_epoch` (shape target only) |
| Mixed precision | enabled (AMP + GradScaler) — unlike SOAP, Muon's Newton-Schulz iteration has no float32-only requirement |
| Gradient clipping | max_norm=1.0 |
| Patience | 15 |
| Architecture | `OCRConvNetMuon` (new), 64→128→256→512 plain residual blocks (no SE, no stochastic depth — deliberately simpler than `OCRConvNetWide`), ~9,677,514 params (hand-computed, see "Verification") |
| Label smoothing | 0.05 |

Per-resolution digit sources: same as SOAP above.

### Router — Ranger (RAdam + Lookahead) (new for v3, replaces the AdaBelief gate)

Applies identically to `v3_mnist_router_ranger_{16,28,32,64,128}.py`:

| Setting | Value |
|---|---|
| Optimizer | Ranger = RAdam + Lookahead (`build_ranger_optimizer()`, inlined in each `v3_mnist_router_ranger_*.py`) |
| RAdam | betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4 |
| Lookahead | k=6, alpha=0.5 (published Ranger defaults) |
| Reference LR | 1e-3 @ REFERENCE_BATCH_SIZE=256, scaled **linearly, capped at 4x** (not sqrt — see "Router training data and optimizer" above for why this differs from the gate's AdaBelief scaling) |
| Batch size | auto-detected per run — not fixed |
| LR schedule | **fixed** 100-step warmup (not steps-per-epoch-derived) → cosine decay, `common/scheduler.py`, `eta_min=1e-4`; `total_steps = 100 × steps_per_epoch` (shape target only) |
| Mixed precision | enabled (AMP + GradScaler) |
| Gradient clipping | max_norm=1.0 |
| Patience | 15 |
| Architecture | `OCRRouterNet` (adapted from v2's `OCRGateNet` — same conv stem, 16→32→64→64), output width changed 2→3, ~62.7K params |
| Output classes | 3-class: digit (0) / UC (1) / LC (2). **`[UNK]` is not a trained class** — produced when top-class softmax confidence < `ROUTER_UNSURE_FLOOR` |
| `ROUTER_UNSURE_FLOOR` | **0.6** (starting-point value, not real-world-tuned — see "Verification") |
| Training data | Digit class: same merged digit-ensemble sources as the digit models, per resolution tier (`digit_sources_for_tier()`). UC/LC classes: EMNIST Balanced, byclass 10-61 only (byclass 0-9 — Balanced's own digit portion — intentionally unused; see "Router training data" above) |

Per-resolution digit-class sources: same as the digit ensemble above
(16×16 = USPS only via `load_base_usps()`; 28×28 excludes USPS; 32/64/128
include it) — the router's UC/LC data (EMNIST Balanced) does not vary by
resolution tier beyond the image resize itself.

---

## File change list

**New shared modules (`common/`) — infrastructure only, no optimizer
algorithms (see "Naming and modularization-scope correction" above):**
`common/__init__.py`, `common/seeding.py`, `common/telemetry.py`,
`common/batch_sizing.py`, `common/checkpointing.py`,
`common/cli_logging.py`, `common/onnx_export.py`, `common/scheduler.py`,
`common/amp.py`

**Rebuilt (adapted from a named v2 file):**
- `supplementary_data.py` — adapted from v2's `supplementary_data.py`
  (extended, not replaced: `digit_sources_for_tier()` and
  `load_base_usps()` added, `BALANCED_TO_BYCLASS` fixed; every existing
  loader/wrapper class otherwise unchanged).
- `ocr_pipeline_mnist.py` — adapted from v2's `ocr_pipeline_mnist.py`
  (router replaces the gate throughout; `short_model_name()` regex
  extended; `MIN_ASPECT_RATIO` lowered from 0.1 to 0.06; box-detection
  docstring extended with the letter-shape review — see above).

**New digit training scripts (15; each adapted from its same-optimizer,
any-resolution v2 counterpart — hyperparameters confirmed identical
across v2 resolution variants, see "What was verified" above):**
- `v3_mnist_digit_soap_16.py`, `v3_mnist_digit_soap_28.py`,
  `v3_mnist_digit_soap_32.py`, `v3_mnist_digit_soap_64.py`,
  `v3_mnist_digit_soap_128.py` — adapted from v2's
  `mnist_soap_32/64/128.py` (16×16 and 28×28 are both new, no v2
  counterpart — SOAP was never trained below 32×32 in v2).
- `v3_mnist_digit_adamw_16.py`, `v3_mnist_digit_adamw_28.py`,
  `v3_mnist_digit_adamw_32.py`, `v3_mnist_digit_adamw_64.py`,
  `v3_mnist_digit_adamw_128.py` — adapted from v2's
  `mnist_adamw_32/64/128.py` (16×16 and 28×28 both new, no v2 counterpart).
- `v3_mnist_digit_muon_16.py`, `v3_mnist_digit_muon_28.py`,
  `v3_mnist_digit_muon_32.py`, `v3_mnist_digit_muon_64.py`,
  `v3_mnist_digit_muon_128.py` — new optimizer, no v2 counterpart;
  architecture and training-loop structure follow the same
  `run_training()`/`main()`/argparse/`_Tee` pattern every other v3 digit
  script uses.

**New router training scripts (5; adapted from v2's single
`mnist_digit_gate.py`, one per resolution tier — the v2 gate trained a
single fixed 64×64 only):**
`v3_mnist_router_ranger_16.py`, `v3_mnist_router_ranger_28.py`,
`v3_mnist_router_ranger_32.py`, `v3_mnist_router_ranger_64.py`,
`v3_mnist_router_ranger_128.py`

**Explicitly removed from the active codebase (present in v2, not
carried into v3 in any form):**
`mnist_adahessian_32.py`, `mnist_adahessian_64.py`,
`mnist_adahessian_128.py`, `mnist_lion_32.py`, `mnist_lion_64.py`,
`mnist_lion_128.py`, `mnist_sgd_32.py`, `mnist_sgd_64.py`,
`mnist_sgd_128.py`, `mnist_digit_gate.py` (superseded by the 4 router
scripts above — see "Router wiring decision"). `01_install_cuda.bat`
(NVIDIA driver/CUDA/interpreter layer, not Python packages) was not part
of this restructure's required output set and was not rebuilt — v3's
package list is unaffected by it either way. `setup_packages.py` was
initially left out for the same reason, then added afterward — see
"Dependency verification tooling added" below.

**New documentation (never existed in v2 — see each file's own header):**
`v3_CHANGELOG.md` (this file), `README.md`, `requirements.txt`,
`setup_packages.py`
