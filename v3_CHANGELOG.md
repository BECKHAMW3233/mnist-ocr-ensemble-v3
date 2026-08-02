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
  (96→192→384→768 feature pyramid, ~4.6M params — **[2026-07-28 erratum:
  wrong, actual ~7.6M — see that date's entry]**), SGD →
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
| Architecture | `OCRConvNetTriplePyramid`, 96→192→384→768, ~4.6M params — **[2026-07-28 erratum: wrong, actual ~7.6M — see that date's entry]** |
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

---

## 2026-07-27 — README: add current on-disk contents section

### What changed

Added a new "Current on-disk contents (as of this writing)" subsection
to `README.md`, directly under the existing projected repository-structure
tree. It lists the actual current state of the repo — every real file
and folder — including `historical_data/claude_code_prompt.md` (an
archived prompt from an earlier session; previously undocumented in the
README) and the two currently-empty `v3_mnist_router_ranger_16/` and
`v3_mnist_router_ranger_28/` output folders. `__pycache__/` directories
(repo root and `common/`, 10 `.pyc` files total) are deliberately
omitted from the new section — auto-generated bytecode cache, not
source — per William's explicit choice when asked to confirm.

### Why

The existing tree in `README.md` was explicitly labeled a *projected*
future layout only ("once every model has been trained and exported")
and didn't reflect what's actually on disk — it omitted `historical_data/`
and the two empty router-output folders entirely. William asked for the
README to be updated to reflect every folder and file actually present
in the project.

### Source

Direct instruction from William (this conversation). Verified against an
actual recursive directory listing of `E:\mnist_v3` (PowerShell
`Get-ChildItem -Force -Recurse`) rather than assumed from the existing
README text, and cross-checked `README.md` and this changelog with a
search for "historical_data", "claude_code_prompt", and "__pycache__" to
confirm neither was already documented anywhere. No external
documentation applies to this change — it's a repo-structure accuracy
fix, not a behavioral or dependency change.

---

## 2026-07-28 — Fix: batch-size probe crashed instantly on Schedule-Free AdamW

### What changed

`common/batch_sizing.py`'s `_probe_batch_size()` now calls
`probe_optimizer.train()` right after constructing the probe optimizer,
guarded by `hasattr(probe_optimizer, "train")`:

```python
probe_optimizer = build_optimizer(model_test.parameters())
if hasattr(probe_optimizer, "train"):
    probe_optimizer.train()
probe_scaler = build_grad_scaler(device, use_amp) if use_amp else None
```

### Problem / why

A real run of `v3_mnist_digit_adamw_16.py` (first actual GPU/PyTorch run
of any AdamW script — see `v3_CHANGELOG.md`'s "Verification" section
above, which already flagged that no real run had happened for any v3
script yet) crashed immediately during batch-size auto-detection, before
any real training step:

```
Exception: Optimizer was not in train mode when step is called. Please
insert .train() and .eval() calls on the optimizer. See documentation
for details.
```

Root cause, confirmed by reading the actual installed
`schedulefree/adamw_schedulefree.py` source directly (not assumed from
this changelog's own prior description of the requirement):
`AdamWScheduleFree.__init__` sets `train_mode=False` by default on every
new instance, and `.step()` unconditionally raises if `train_mode` is
still `False`. `_probe_batch_size()` builds a fresh, throwaway
`probe_optimizer` via the caller's `build_optimizer(...)` to test each
candidate batch size, then runs it through `common/amp.py`'s
`amp_train_step()` — which is correctly optimizer-agnostic and has no
Schedule-Free-specific knowledge — without ever calling `.train()` on
that probe optimizer first. The real training loop in each
`v3_mnist_digit_adamw_*.py` script already calls `optimizer.train()`
correctly on its own (real, non-probe) optimizer instance before its
real epoch loop; only the shared probe path was missing the equivalent
call.

**Why a plain, unconditional `.train()` call would have been wrong:**
`common/batch_sizing.py` is shared by all 20 training scripts. SOAP,
Muon, and Ranger are all standard `torch.optim.Optimizer` subclasses
with no `.train()`/`.eval()` concept at all — calling `.train()`
unconditionally on their probe optimizer would raise `AttributeError`
for all three. The `hasattr(...)` guard makes this a no-op for anything
that doesn't define `.train()`, so this fix doesn't touch SOAP/Muon/
Ranger's behavior at all. An `isinstance(..., AdamWScheduleFree)` check
was considered and rejected — it would have forced `common/
batch_sizing.py` (imported by every script, including SOAP/Muon/Ranger,
none of which need the `schedulefree` package) to import `schedulefree`
just for this one type check, adding an unwanted hard dependency for 15
of the 20 scripts that don't use that optimizer.

**Scope:** `common/batch_sizing.py` is imported by all 5
`v3_mnist_digit_adamw_{16,28,32,64,128}.py` scripts (confirmed via
`grep` — each imports `determine_batch_size`/`cap_batch_size_for_min_steps`
from `common.batch_sizing`, none has a local copy of
`_probe_batch_size()`), so this one fix resolves the crash for all five
resolutions, not just the 16×16 script that actually hit it first.

### Source

Direct instruction from William (this conversation, after the real
`v3_mnist_digit_adamw_16.py` crash traceback he pasted). Root cause
verified by reading the actual installed
`C:\Users\Will\AppData\Local\Python\pythoncore-3.14-64\Lib\site-packages\schedulefree\adamw_schedulefree.py`
directly — specifically `__init__`'s `train_mode=False` default and
`step()`'s `if not self.param_groups[0]['train_mode']: raise` check —
not assumed from memory or from this changelog's own earlier
description of the Schedule-Free train/eval requirement. Scope (which
scripts share this code path) verified via `grep` across
`v3_mnist_digit_adamw_*.py`, not assumed from the modularization
description alone.

---

## 2026-07-28 — Fix: real training OOM'd on first step despite GB of "free" VRAM

### What changed

`common/__init__.py` now sets, at package-import time:

```python
import os
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
```

This runs before `import torch` in every script — each script's first
import is `from common.seeding import ...`, which triggers this
package's `__init__.py` first. `setdefault()`, not a hard set, so an
explicit value already set in the caller's own shell isn't overridden.

### Problem / why

After the previous entry's batch-size-probe fix, `v3_mnist_digit_adamw_28.py`
got all the way through batch-size auto-detection (17 candidates,
refined to 4864) but OOM'd on the very first real training step, inside
`OCRConvNetWide.forward()`'s `stage2`:

```
[W...] CUDACachingAllocator.cpp:3933] memory allocation failed with OOM
on device 0 while trying to allocate 979369984 bytes
(free: 5502926848, total: 17170956288).
```

repeated identically for ~50-90 seconds (William had to Ctrl+C both
times, twice reproducing the exact same numbers byte-for-byte on two
independent process launches) before it either exhausted retries or was
interrupted. Note: the resulting `Exception ignored in atexit callback
... dump_compile_times ... KeyboardInterrupt` noise on Ctrl+C is
harmless — `NUM_WORKERS=18` worker processes all receiving the interrupt
at once and racing each other during shutdown, not a separate bug.

The confusing part: ~5.1-5.5 GB was reported "free," far more than the
~934 MB request that failed — not a real capacity shortage. Root cause,
confirmed via PyTorch's own current devlog on the CUDA caching allocator
(dated 2026-06-01, matching PyTorch 2.13.0, the version installed here):
without `expandable_segments`, each differently-sized allocation can
land in its own `cudaMalloc` segment, and "blocks in different segments
can never merge" — `torch.cuda.empty_cache()` cannot merge or reclaim
across that boundary. `common/batch_sizing.py`'s batch-size probe
(`determine_batch_size()`/`_probe_batch_size()`) deliberately allocates
and frees 17 different tensor sizes in the same process, including 5
sizes it intentionally pushes to failure during binary-search refinement
(8192, 6144, 5120, 4992, 4928, 4896, 4880, 4872 all failed by design) —
exactly the repeated-different-sizes pattern the devlog describes as
fragmenting the allocator. By the time real training's first forward
pass needed one large contiguous ~934 MB block, no single free segment
was big enough, even though the sum of free memory was several GB.

**Where the fix lives, and why not elsewhere:** considered setting this
in each script directly (matching the existing
`apply_cublas_workspace_config()`-before-`import torch` convention in
`common/seeding.py`), but that would mean editing all 20 scripts for a
2-line change. `common/__init__.py` runs automatically on first `common.*`
import — already the first import in every script — so one file covers
all 20 with no per-script call needed.

**Caveat surfaced to William before he approved this (not hidden):**
PyTorch's devlog notes `expandable_segments` has had unresolved CUDA IPC
compatibility issues historically. `common/distributed.py` (multi-GPU
DDP) is already flagged in its own docstring and elsewhere in this
changelog as not yet validated on real multi-GPU hardware — this change
doesn't make that any less validated, but is worth remembering if DDP is
ever tested on real multi-GPU hardware and something IPC-related acts
up. Every run tested so far is single-GPU, which this caveat doesn't
affect.

### Source

Direct instruction from William (this conversation), after two real,
byte-for-byte-reproducible crash tracebacks he pasted. Root cause and
fix verified via PyTorch's CUDA caching allocator devlog
(https://docs.pytorch.org/docs/devlogs/eager/2026-06-01-cuda-caching-allocator/)
and cross-checked against the actual current PyTorch 2.13 docs
(https://docs.pytorch.org/docs/2.13/notes/cuda.html) specifically to
confirm `PYTORCH_ALLOC_CONF` is the current primary env var name for
this installed version (`PYTORCH_CUDA_ALLOC_CONF` is kept only as a
backward-compatible alias) — not assumed from general/possibly-stale
knowledge of this flag's name. Import-order timing (this needs to run
before `import torch`) verified by reading `v3_mnist_digit_adamw_28.py`'s
actual import block and `common/seeding.py`'s own documented
`apply_cublas_workspace_config()`-before-`import torch` precedent, not
assumed. Fix placement (one file vs. all 20 scripts) was presented to
William as an explicit choice before writing anything; he chose the
one-file `common/__init__.py` approach.

---

## 2026-07-28 — Follow-up: expandable_segments was a no-op on this platform; real cause was too little safety margin

### What was found

The retried `v3_mnist_digit_adamw_28.py` run showed
`UserWarning: expandable_segments not supported on this platform` at
startup — the env var from the previous entry was read correctly (that's
why PyTorch warned about it at all), but this specific PyTorch Windows
build doesn't support the CUDA driver VMM API expandable_segments
depends on, so it silently did nothing. Verified via web search: this is
a known, documented Windows limitation (`PYTORCH_C10_DRIVER_API_SUPPORTED`
not defined in this build), not a mistake in how the env var was set.

Training then OOM'd again, but the new exception's own numbers didn't
support fragmentation as the (main) cause: `10.36 GiB allocated` vs. only
`21.01 MiB reserved but unallocated` — fragmentation shows up as a
*large* reserved-but-idle figure; this was tiny. 10.36 GiB was genuinely
in active use.

**Narrowed down with a real controlled test, per William's own request
("basic fixes first, then narrow it down"):** retried with
`--batch-size 2048` (a size already confirmed working in the auto-
detector's own coarse pass, well below the auto-detected-and-failing
4864) and let it run 5 full epochs. Result: completely healthy —
`vram_peak_alloc_gb` flat at 5.4 GB every epoch, `vram_peak_reserved_gb`
stepped up once (6.8 GB → 8.3 GB, epoch 1 → 2) then held perfectly flat
for epochs 2-5. No leak. But extrapolating that 8.3 GB peak linearly to
batch 4864 (2.375× larger, and CNN activation memory scales close to
linearly with batch size) gives ≈19.7 GB — more than this 16 GB card has
at all. 4864 was very likely never sustainable here, not just "close to
the edge."

This also means an earlier take in this same conversation was wrong:
when William first suggested probing with more steps and real data,
Claude pushed back, reasoning a fixed batch size stabilizes within a
step or two since `cudnn.benchmark=False` (set in `common/seeding.py`).
The 2048 data contradicts that — the reserved-memory jump happened
between epoch 1 and epoch 2, not within a handful of steps. William's
instinct was right; recorded here so this isn't quietly glossed over.

### What changed

1. `common/__init__.py` — `PYTORCH_ALLOC_CONF` now also sets
   `garbage_collection_threshold:0.8` (proactively reclaims cached-but-
   idle GPU memory once usage crosses 80%, instead of only reclaiming
   when forced). Verified against the current PyTorch 2.13 docs (same
   page as above). Honestly flagged to William before he approved it:
   this addresses a *different* failure mode (idle memory being
   hoarded) than the one actually observed here (genuine active use) —
   it's a real, useful safeguard for the future, not a fix for what
   already happened.

2. `common/batch_sizing.py` — new `VRAM_RESERVE_GB = 3.0` constant
   (previously an unnamed, scattered `1024` MB / `1.0` GB literal in
   three places), and all three of `_probe_batch_size()`'s decisive
   VRAM checks (the nvidia-smi-based per-step check, its subprocess-
   failure fallback, and the final post-loop verdict) now check
   `torch.cuda.max_memory_reserved()` instead of `memory_allocated()`.
   Reserved is the more conservative figure (always ≥ allocated, and
   it's what actually blocks new allocations) — and the 2048 run showed
   a real ~2.9 GB reserved-vs-allocated gap even when everything was
   healthy, so checking allocated with only a 1 GB margin left a real
   blind spot. `3.0` was chosen to clear that observed gap with some
   margin left over; presented to William as a specific, named number
   before writing, per this project's own convention for batch-size-
   related constants.

### Source

Direct instruction from William (this conversation), following his own
`--batch-size 2048` test and the 5-epoch transcript he pasted back
(`v3_mnist_digit_adamw_28_cli_20260728_012623.txt`). `expandable_segments`
Windows limitation verified via web search (PyTorch GitHub issues
referencing `PYTORCH_C10_DRIVER_API_SUPPORTED`). `garbage_collection_threshold`
verified against the current PyTorch 2.13 CUDA docs
(https://docs.pytorch.org/docs/2.13/notes/cuda.html), same page already
cited in the previous entry. `VRAM_RESERVE_GB`'s value and the
allocated-vs-reserved reasoning came from directly reading
`common/telemetry.py` (to confirm what the console's `VRAM X/Y GB`
figures actually are — peak allocated / peak reserved) and from the
real 2048-batch transcript's own per-epoch numbers, not assumed.

---

## 2026-07-28 — README: refresh current-on-disk-contents (5 models now complete)

### What changed

Rewrote `README.md`'s "Current on-disk contents" section to match a
fresh recursive directory listing. Since the last version of this
section was written, 5 of the 20 models completed full runs
(`v3_mnist_digit_soap_16`, `v3_mnist_digit_adamw_16`,
`v3_mnist_digit_muon_16`, `v3_mnist_router_ranger_16`,
`v3_mnist_router_ranger_28` — each now has `_best.pt`, `_final.pt`,
`.onnx`, `_log.csv`, `_curves.png`, and a CLI transcript) and
`v3_mnist_digit_adamw_28` is mid-run (`_best.pt` + `_resume.pt` +
transcript, no final export — stopped via Ctrl+C per William, resumable).
Also noted `.claude/settings.local.json`, which now exists on disk
(Claude Code's own local session settings) — excluded from the tree on
the same basis as `__pycache__/`: a tooling artifact, not project
content.

### Why

William asked for the README to reflect all the new files that had
appeared in the folders since the last pass, in the same message that
approved the two batch-sizing fixes above.

### Source

Direct instruction from William (this conversation). Verified against a
fresh recursive listing of `E:\mnist_v3` (`find`, excluding
`__pycache__`) rather than assumed from the previous README version or
from memory of earlier conversation turns.

---

## 2026-07-28 — Correction: the actual fix for `v3_mnist_digit_adamw_28`'s crash was a Windows page-file change, not the VRAM entries above

### What this corrects

The two entries directly above this one (`garbage_collection_threshold` /
`VRAM_RESERVE_GB`) are real fixes for a real, separately-confirmed GPU-VRAM
problem (batch size 4864 genuinely didn't fit — confirmed via a clean,
non-resumed `--batch-size 4864` test that still OOM'd with 10.79 GiB
genuinely allocated). They are being kept, not reverted. But William
correctly pushed back that they weren't the actual explanation for why
this script kept crashing, and he was right: after those fixes, a retry
at the auto-detector's new (smaller) batch size of 3752 got past the
GPU-side warnings — which turned out to be transient and recoverable,
real training visibly progressed for several batches — and then crashed
on a completely different, non-GPU error:

```
RuntimeError: Couldn't open shared file mapping: <torch_22980_...>, error code: <1455>
```

This is the exact same error, same code, same mechanism as the one
diagnosed for `v3_mnist_router_ranger_28` earlier in this same session
(see that crash's own diagnosis above in this changelog): Windows error
1455 = `ERROR_COMMITMENT_LIMIT` — a **system RAM / page-file** ceiling
being hit in the DataLoader worker → main-process shared-memory handoff,
via `NUM_WORKERS=18` each holding its own full duplicated copy of the
merged dataset (confirmed cause for the router: Windows `multiprocessing`
uses `spawn`, not `fork`, so there's no copy-on-write sharing across
worker processes). Not GPU VRAM at all — a completely separate resource
ceiling that happened to produce crashes in the same training runs the
VRAM fixes were also touching, which is what made this easy to
conflate.

### What actually fixed it

William increased the Windows page file (`D:\pagefile.sys`) from a fixed
8 GB to a custom 16 GB initial / 32 GB maximum, directly in Windows
(Advanced system settings → Performance → Advanced → Virtual Memory) —
an OS-level system setting, not a project file, so there is no code diff
for this entry. This raises the system commit ceiling from ~71 GB
(63.2 GB RAM + 8 GB page file) to ~79-95 GB, comfortably clearing the
~31 GB confirmed overhead from `NUM_WORKERS=18`'s dataset duplication.
Reported working by William after this change and a retry.

**`NUM_WORKERS` reduction was discussed as the alternative/complementary
fix** (directly cuts the per-worker duplication instead of raising the
ceiling around it) but wasn't needed once the page file was increased.
Not applied — recorded here as the fallback option if this recurs,
since `NUM_WORKERS` is explicitly sensitive in this project (see this
file's own project history) and shouldn't be changed without being asked
again specifically.

### Source

Direct instruction from William (this conversation) and his own real
crash tracebacks (three retries of `v3_mnist_digit_adamw_28.py`, pasted
in full) plus his report that the page-file change fixed it. Root cause
verified by direct comparison against the router's own already-verified
diagnosis earlier in this changelog (same error string, same code 1455,
same Microsoft Win32 error code reference already cited there), not
re-derived from scratch. No new external documentation beyond what was
already cited for the router's identical error.

---

## 2026-07-28 — `VRAM_RESERVE_GB` adjusted from 3.0 to 1.5

### What changed

`common/batch_sizing.py`'s `VRAM_RESERVE_GB` changed from `3.0` to `1.5`,
per William's direct instruction. Still checked against peak
`max_memory_reserved()`, not just allocated — only the number changed,
not the mechanism from the earlier entry.

### Why

William's reasoning, stated directly: with the page file now enlarged
(previous entry), the system-RAM/page-file crash class is covered
separately, so this constant only needs to answer the GPU-VRAM question
— and `3.0` was wider than he wanted given his own history of `1.0`
working fine (at a lower `NUM_WORKERS`, before this project's
auto-scaling raised it to 18, and before the page file was enlarged).
`1.5` is a deliberate middle ground between the original `1.0` (which
let batch 4864 through despite a confirmed real OOM) and `3.0` — his
own call to make, and he asked to "let it ride from there" and adjust
again if a real run still OOMs at this setting.

### Source

Direct instruction from William (this conversation) — no external
source; this is a judgment call on a project-specific constant, not a
documentation-verifiable fact.

---

## 2026-07-28 — Confirmed: `VRAM_RESERVE_GB=1.5` + enlarged page file resolved `v3_mnist_digit_adamw_28`

Real run, `v3_mnist_digit_adamw_28_cli_20260728_020307.txt`: auto-detector
landed on batch 3744, then 6 full epochs completed cleanly — no GPU OOM,
no DataLoader shared-memory errors, no warnings of any kind. VRAM held
flat at 9.8/13.0 GB and system RAM flat at 30.2-30.6/63.2 GB across every
epoch (the same healthy "one step up, then flat" pattern seen in every
other successful run this session). Val accuracy climbed 98.5% → 99.58%
over the 6 epochs. William stopped it deliberately via Ctrl+C once
satisfied it was stable, not because of a crash.

This closes out the multi-part crash investigation recorded across the
several entries above: the GPU-VRAM margin (`VRAM_RESERVE_GB`) and the
system-RAM page-file ceiling were two separate, both-real problems, and
both are now addressed — the former in code, the latter as a Windows
setting outside this repo. No further action needed unless a future run
hits either failure mode again.

Source: William's own transcript, read directly, not summarized from his
description.

---

## 2026-07-28 — `VRAM_RESERVE_GB` reverted from 1.5 to 1.0, per repeated direct instruction

### What changed

`common/batch_sizing.py`'s `VRAM_RESERVE_GB` changed from `1.5` back to
William's original `1.0`.

### Why

William asked for this twice, directly. Before making the change, William
asked Claude to pull and compare every training transcript produced so
far across all resolutions/optimizers. That comparison surfaced a real
methodological problem with part of the evidence the `1.5`/`3.0` changes
had leaned on: `grep -c "reset_peak_memory_stats"` across all 20 scripts
confirmed `v3_mnist_digit_adamw_*.py` and `v3_mnist_router_ranger_*.py`
**never** call `torch.cuda.reset_peak_memory_stats()` per epoch, while
`v3_mnist_digit_soap_*.py` and `v3_mnist_digit_muon_*.py` do. This means
adamw's and router's console `VRAM X/Y GB` telemetry is a cumulative
peak since the batch-size probe ran, not a clean per-epoch figure — so
the "13.0 GB peak" cited as evidence for a wider margin in earlier
entries may have been inflated by leftover probe-candidate peaks, not
real steady-state usage. Separately, a real `v3_mnist_digit_adamw_16.py`
run (batch 11528, before tonight's fixes) held a stable 12.3 GB peak
with the *original* 1.0 GB margin and had no issue, which is real,
directly-comparable evidence for reverting.

**Not reverted or explained away:** the clean, non-resumed
`--batch-size 4864` test's `torch.OutOfMemoryError` (`10.79 GiB is
allocated by PyTorch`) is unaffected by the telemetry-reset issue above —
that number came from the live CUDA allocator via PyTorch's own
exception text at the moment of failure, not from this project's
per-epoch logging. That crash was real. Reverting to `1.0` is William's
explicit decision to accept the risk of hitting it again in exchange for
not being unnecessarily conservative everywhere else, given the auto-
detector will simply pick smaller batches on future OOMs.

**Also noted, unresolved:** `v3_mnist_router_ranger_64.py`'s transcript
showed a `VRAM` reading of `9.8/18.9GB` — 18.9 GB exceeds this card's
15.99 GiB usable capacity, which shouldn't be possible even accounting
for the cumulative-peak issue above. Not fully explained; possibly
Windows WDDM backing some allocation with shared/system memory rather
than dedicated VRAM, but this is speculation, not confirmed, and is
flagged here rather than asserted as fact.

### Source

Direct, repeated instruction from William (this conversation).
`reset_peak_memory_stats()` presence/absence verified via `grep -c`
across all 20 training scripts, not assumed. The adamw_16/12.3GB
comparison came from directly reading
`v3_mnist_digit_adamw_16_cli_20260728_005856.txt`, not recalled from
memory.

---

## 2026-07-28 — README: all prior training output cleared; every model to be redone on current code

### What changed

Rewrote `README.md`'s "Current on-disk contents" section again. Every
output directory that previously held a completed or in-progress run
(`soap_16`, `soap_28`, `soap_64`, `adamw_16`, `adamw_28`, `muon_16`,
`muon_28`, `router_ranger_16`, `router_ranger_28`, `router_ranger_64`)
is now empty — confirmed via `ls -la` on each directory, not assumed.
10 of the 20 scripts have an empty output directory (created by at
least one prior run attempt); the other 10 (`*_32.py`, `*_64.py`/
`*_128.py` variants not listed above) have never been run at all — no
directory exists for them.

### Why

William's position, stated directly: a trained model is only valid if
it was trained on the current state of the code. Several models
finished earlier in this session, before some of tonight's shared-
infrastructure fixes (`common/batch_sizing.py`'s probe fix and margin
changes, `common/__init__.py`'s `PYTORCH_ALLOC_CONF`) existed. Rather
than track which specific fix each old run predated, William cleared
all prior output and will redo everything on the current, stable code
going forward. (Claude's own read, given directly to William beforehand:
none of tonight's changes actually alter training math/gradients/
weights — they only affect batch-size selection and CUDA allocator
internals — so the already-completed runs weren't technically invalid.
This is recorded as William's explicit methodology choice — consistent
code version across every run — not a correction of a real defect in
the old runs.)

### Source

Direct instruction from William (this conversation). Empty-directory
state verified via `ls -la` on each of the 10 existing output
directories, and via a fresh `find` across the whole repo confirming
which of the other 10 scripts have no directory at all — not assumed
from the earlier (now-stale) README version.

---

## 2026-07-28 — Add `run_all_training.ps1` (new file)

### What changed

New file: `run_all_training.ps1`, a PowerShell script that runs all 20
v3 training scripts in sequence (16x16 → 28x28 → 32x32 → 64x64 → 128x128,
each resolution as soap → adamw → muon → router). Skips any script whose
output directory already has a `_final.pt` (already trained to
completion); everything else is launched in order and waits for each to
finish before starting the next.

Does not add any new resume/state-tracking mechanism — deliberately
reuses what already exists: a script with a `_resume.pt` but no
`_final.pt` gets re-launched and resumes on its own via that script's
existing checkpoint/resume logic (see `common/checkpointing.py` and this
changelog's own entries on how that works), so Ctrl+C at any point during
a run behaves exactly as it already does for a manually-run script — no
new corruption risk, confirmed by reading `common/checkpointing.py`
directly before answering William's question about it (checkpoints only
ever save at the end of a fully-completed epoch, never mid-epoch).

### Why

William asked to be able to run training "until I say stop" and resume
later "later in the day or week," across all 20 scripts, without manually
relaunching each one. Per this project's Testing rule, Claude does not
run training itself, including via a script — this file is something
William runs and controls himself; Claude only wrote it.

### Source

Direct instruction from William (this conversation) — design (ordering,
the `_final.pt`-based skip check, reusing existing resume logic instead
of building new state tracking) proposed by Claude and confirmed with
William before writing anything, per this project's standard diff-then-
approve sequence.

---

## 2026-07-28 — Documentation audit and fix pass across the entire codebase

William asked for a full verification pass on every comment/docstring in
every project file — clear, accurate, and cited where a technical claim
is made — followed by "fix them all please." This was executed as 6
parallel read-only review agents (findings reported to William in full
before any fix), then 6 parallel fix agents given exact, pre-specified
edits (not open-ended review), with every resulting diff verified by
Claude directly (re-read/grepped) before being logged here. Several of
the fix agents independently identified and refused to write without
William's own explicit chat approval — correct behavior per this
project's rules, since a task prompt from another agent isn't the same
as William's own "go ahead"; William had, in fact, already given that
approval in chat, so Claude applied the pre-verified diffs directly
rather than re-running the agents. The entries below cover what actually
changed, grouped by file family to keep this readable rather than 26
separate single-line entries.

### v3_mnist_digit_muon_{16,28,32,64,128}.py (3 fixes × 5 files)

**What:** (1) Fixed a self-contradictory warmup description — said
"500-step-equivalent" while the same sentence cited `WARMUP_STEPS=300`
and said it "differs from SOAP's 500"; changed to "300-step". (2) Fixed
a citation that trimmed its own source's date range — "2025 independent
benchmarks" → "2025-2026 independent benchmarks", matching
v3_CHANGELOG.md's actual roster-change entry. (3) Fixed a dangling "See
Part 1 of the v3 restructure" reference (no section anywhere in this
file is literally labeled "Part 1") to name the real section directly:
"v3_CHANGELOG.md's Resolution ladder split section".

**Why:** All three were confirmed factually wrong or unresolvable
against this changelog's own content, not style preferences.

**Source:** Direct instruction from William (fix-everything approval),
executed against findings from a dedicated review pass. Each claim was
cross-checked against v3_CHANGELOG.md directly (grep for "Part 1",
"500-step"/"300-step" in the Muon standard-settings table, and the
roster-change entry's actual date range) before being changed — not
assumed from the review pass alone.

### v3_mnist_digit_soap_{16,28,32,64,128}.py (4 fixes × 5 files)

**What:** (1) Fixed a wrong parameter count — docstrings said "~4.6M
parameters"; independently hand-computed from the actual
`OCRConvNetTriplePyramid`/`TripleBlock`/`SEBlock` architecture
(stem + 4 stages + 5-layer classifier head), twice, by two different
agents, both landing on 7,573,482 (~7.6M) — corrected to "~7.6M
parameters". (2) Fixed false "Adapted from v2" lineage claims:
`soap_16.py`/`soap_28.py` claimed adaptation from v2 files
(`mnist_soap_16.py`/`mnist_soap_28.py`) that never existed — v2's SOAP
tiers were only 32/64/128 — reworded to state plainly these are new for
v3 with no v2 counterpart. `soap_32/64/128.py` correctly claim v2
adaptation but their trailing clause named only 16x16 as new,
implying 28x28 wasn't — fixed to name both. (3) Same "Part 1" fix as
Muon, in this file and in `supplementary_data.py:1126` (a 6th occurrence
found during a final project-wide sweep, not caught by the original
review pass). (4) Softened the uncited "SOAP requires float32" claim
(stated 3x per file) — a review agent checked the actual
`pytorch_optimizer` SOAP source directly and found it already handles
float32 casting internally regardless of caller precision, so "requires"
overstated certainty about why AMP is disabled here. Reworded the two
docstring instances to "kept disabled for SOAP, per v2 practice — not
independently re-verified against pytorch_optimizer's current
internals"; left the short runtime print statement as-is (terse console
output, not a documentation claim).

**Why:** All four are real defects — one factual error (params), one
provenance error confirmed false against this project's own file-change
list, one dead citation, one overstated-certainty claim confirmed
against the actual upstream library source.

**Note, not corrected:** `v3_CHANGELOG.md` itself repeats the same wrong
"~4.6M" figure twice (its own standard-settings table and an earlier
entry). Per this project's own rule ("never edit or delete a past
changelog entry... add a new entry correcting it"), those are left
untouched — this paragraph is that correction. The real figure is
~7.6M (7,573,482), per the independent hand-computation above.

**Source:** Direct instruction from William. Parameter count verified
by direct architecture computation (twice, independently, matching);
v2-lineage claims verified against `v3_CHANGELOG.md`'s File change list;
float32 claim verified against the actual installed `pytorch_optimizer`
package source, not assumed.

### v3_mnist_digit_adamw_{16,28,32,64,128}.py (2 fixes × 5 files)

**What:** Same false "Adapted from v2" pattern as SOAP —
`adamw_16.py`/`adamw_28.py` claimed nonexistent v2 predecessors
(v2's AdamW tiers were also only 32/64/128), fixed to state they're new
for v3; `adamw_32/64/128.py`'s trailing clause fixed to name both 16x16
and 28x28 as new instead of only 16x16. Same "Part 1" citation fix as
Muon/SOAP.

**Why:** Confirmed false against `v3_CHANGELOG.md`'s File change list
and "What was verified" section, which only ever reference v2's 32/64
AdamW tiers.

**Source:** Direct instruction from William, verified against
`v3_CHANGELOG.md` directly.

### v3_mnist_router_ranger_{16,28,32,64,128}.py (2 fixes × 5 files)

**What:** (1) Removed a stale `make_dataloader()/` reference — that
function doesn't exist in any router file (only in the digit-ensemble
scripts, confirmed via grep both ways), copy-paste residue from the
digit-script template; the router files' real function is
`make_train_loader()`, which the comment already also named. (2) Fixed
the same "Part 1" dangling citation as the other three optimizer
families — present in all 5 router files at a location the original
review pass did not catch; found during a final project-wide sweep
after the rest of the fix pass completed, and fixed at that point.

**Why:** (1) is a genuine dead reference — a reader following it finds
nothing. (2) is the same confirmed-unresolvable citation as elsewhere
in this entry.

**Source:** Direct instruction from William. (1) confirmed via grep
showing zero `make_dataloader` definitions in any router file and
exactly one in each digit-ensemble file. (2) found via a full
project-wide grep for the same pattern already fixed elsewhere, run
after the six fix agents reported back, specifically to catch anything
they'd missed.

### common/__init__.py, common/batch_sizing.py, common/scheduler.py, common/telemetry.py, supplementary_data.py (10 fixes)

**What:**
- `common/__init__.py`: docstring said "No behavior changes were made
  while extracting" — no longer accurate now that this file also
  contains the `PYTORCH_ALLOC_CONF` hotfix (a real behavior change, per
  this changelog's own earlier entries); added a sentence distinguishing
  the original behavior-neutral extraction from the later hotfix. Also
  added a note that `expandable_segments:True` is a confirmed no-op on
  this Windows PyTorch build (per this changelog's own earlier finding),
  so a future reader doesn't assume both halves of the setting are
  active.
- `common/batch_sizing.py`: docstring said the candidate ladder tops out
  at "4096" — actual code (and a real logged run that reached it) goes
  to 16384; fixed. Added a clarifying comment on the previously-bare
  `if _free_vram < 1.5:` pre-flight check, noting it's independent of
  `VRAM_RESERVE_GB` below it, since that constant already has 14 lines
  of hard-won history and the unexplained `1.5` sitting next to it
  invited confusion.
- `common/scheduler.py`: `WarmupCosineScheduler`'s `eta_min` has
  different semantics than PyTorch's own same-named `CosineAnnealingLR`
  parameter — it's a fractional multiplier of each group's base LR, not
  an absolute floor (e.g. `eta_min=1e-6` with `base_lr=1e-3` actually
  bottoms out at `1e-9`). Zero comments previously flagged this; added
  one.
- `common/telemetry.py`: one claim about `--query-compute-apps` lacking
  a per-GPU filter on some systems was stated with the same confidence
  as a properly-cited claim three lines above it; marked as an
  observation, not independently verified, to match its actual
  confidence level.
- `supplementary_data.py`: fixed two wrong sample-count numbers, both
  verified against actual dataset documentation —
  `EMNISTDigitsDataset`'s docstring said "280,000 training" (that's the
  *total*; real training count is 240,000); `USPSDataset`'s docstring
  said "9,298 training" (that's the total; real training count is
  7,291) — this same USPS error also appeared a second time, in
  `load_base_usps()`'s own docstring, found and fixed during the final
  project-wide sweep. Fixed the module docstring's "~440,155 raw digit
  samples" figure, which mixed dataset totals and training-only splits
  inconsistently and didn't match what the code actually loads at
  training time — recomputed from consistent training-split figures
  (~388,148) with a note on what's included. Added "specific to the
  original machine" clarification to 4 of 6 hardcoded path constants
  (KAGGLE_DIR, CHARS74K_HND/IMG, PGHWLD_DIR) that previously only
  implied this via a section header 15-35 lines above rather than
  restating it individually.

**Why:** Two are factual number errors (verified externally), two are
self-contradictions between a docstring and the code/file it sits next
to, two are missing-but-warranted clarifying comments (one a real
semantic gotcha, one a citation-confidence mismatch), and the path
constants were an inconsistency-of-emphasis issue, not a missing fact.

**Source:** Direct instruction from William. Dataset sample counts
verified via web search against EMNIST/USPS official documentation, not
assumed. Candidate-ladder and `PYTORCH_ALLOC_CONF`/`expandable_segments`
claims cross-checked directly against this changelog's own prior entries
(the `16384` real-run entry, the `PYTORCH_ALLOC_CONF` hotfix entry, and
the `expandable_segments`-not-supported finding). `eta_min` semantics
traced directly through `WarmupCosineScheduler.step()`'s actual formula,
not assumed from PyTorch's convention.

### ocr_pipeline_mnist.py, setup_packages.py, requirements.txt

**What:** Added documentation for 5 of `run_pipeline()`'s 8 parameters
that had none. Added real citations (OpenCV's morphological-operations
and distance-transform/watershed documentation) to two "standard
technique" claims that previously named no source. Reworded a specific
"~45px stray stroke" measurement and a `SIZE_WINDOWS` threshold example
from stated-as-observed-fact to explicitly-reasoning/estimate framing,
matching this project's own existing honesty caveat used for the
`MIN_ASPECT_RATIO` change two paragraphs later — a review agent found
zero backing for either number anywhere in this changelog, and this
changelog's own Verification section already states no test images were
available in the environment that built this code. Removed a dead
no-op ternary (`axis=1 if False else 0`, always evaluates to `axis=0`).
Added docstrings to 4 previously-undocumented functions
(`normalize_char`, `predict_char_topn`, `merge_nearby_boxes`,
`resolve_image_paths`). Fixed a false citation to "README.md's
Requirements section" for specific version pins (torch 2.13.0 etc.)
that section doesn't actually contain — found in 3 separate locations
across `setup_packages.py` (2 originally targeted, 1 flagged-but-
correctly-left-alone by the fix agent since it was outside its exact
scope) and a 4th, previously-unflagged instance in `requirements.txt`
found during the final sweep; all 4 reworded to point at where the
versions are actually corroborated (this changelog's CUDA-allocator
entry, `run_all_training.ps1`'s hardcoded python.exe path).

**Why:** Undocumented parameters and functions are a clarity gap, not a
correctness bug. The two "standard technique" claims and the two
empirical numbers were real citation gaps — the technique claims are
independently true (real, standard CV techniques) but weren't
attributed the way the rest of this file attributes its sources; the
specific numbers had zero support anywhere and this project's own rules
specifically flag unverified technical claims stated as fact as a
known failure mode to avoid. The dead ternary and false citation are
straightforward defects.

**Source:** Direct instruction from William. The two "standard
technique" claims were confirmed as real, standard CV techniques (not
fabricated) before citing them, just previously uncited. The empirical
numbers' lack of backing was confirmed by grepping this changelog for
every related term (stray stroke, SIZE_WINDOWS, the specific pixel
values) and finding zero matches, plus this changelog's own Verification
section admitting no test images existed in the build environment — not
asserting fabrication, only that nothing in this repository currently
supports those specific numbers. The false README citation was
confirmed by reading README.md's actual Requirements section directly
and finding no version pins there at all.

### Final verification

After all fix agents reported back, Claude ran a project-wide grep for
every confirmed-wrong pattern above (`Part 1 of the v3`, `9,298 train`,
`280,000 training`, `4.6M parameters`, `500-step-equivalent`,
`2025 independent benchmarks`, `README.md's Requirements section`,
the dead ternary, and the stale `make_dataloader()` reference) across
every `.py`/`.txt` file in the repository — zero remaining matches,
except the two intentionally-untouched `v3_CHANGELOG.md` occurrences of
"~4.6M" noted above. This sweep is what caught the 3 fixes the six
per-family review/fix agents missed (router's "Part 1" reference in all
5 files, `supplementary_data.py`'s second USPS-count instance and its
own separate "Part 1" reference, and `requirements.txt`'s copy of the
README citation bug) — confirmed by directly re-reading each, not
assumed from the review agents' own self-reported completeness.

---

## 2026-07-28 — Errata: acknowledged two SOAP parameter-count errors inline

### What changed

`v3_CHANGELOG.md` line 57 and the SOAP row of the "Standard settings
record" table (line ~1186) both stated "~4.6M params" for
`OCRConvNetTriplePyramid` — confirmed wrong (see the "Documentation
audit and fix pass" entry above; real figure is ~7.6M, independently
hand-computed twice). Per William's explicit instruction, added a short
bracketed pointer immediately next to each occurrence —
`**[2026-07-28 erratum: wrong, actual ~7.6M — see that date's entry]**`
— without altering, removing, or reflowing any of the original text
around it.

### Why

William asked for factual errors in past entries to be acknowledged
in-place with a pointer to the correction, explicitly distinguishing
this from rewriting history: the original wrong figure stays exactly as
first written in both spots, fully intact — this only adds a forward
annotation next to it, the same function a published errata slip serves
for a book. This is narrower in scope than "every fix from tonight" —
checked directly, the other numeric fixes from the same audit
(EMNIST/USPS sample counts, the batch-size candidate ladder, etc.) were
never actually asserted wrong in any past `v3_CHANGELOG.md` entry itself
— those were only wrong in the `.py` files' own docstrings, which were
already corrected directly (no changelog erratum needed for those).

### Source

Direct instruction from William (this conversation), executed only
after confirming via `grep` that the "~4.6M" figure was the sole
instance of a past changelog entry itself (not just a `.py` file)
asserting something later confirmed wrong.

---

## 2026-07-29 — README: refresh current-on-disk-contents (9 models now complete)

### What changed

Rewrote `README.md`'s "Current on-disk contents" section (and its
summary counts) to match a fresh recursive directory listing of
`E:\mnist_v3`. Since the section was last written: `v3_mnist_digit_soap_28`
went from mid-run to complete (now has `_final.pt`, `.onnx`, `_log.csv`,
`_curves.png`, plus a second CLI transcript from a rerun);
`v3_mnist_digit_adamw_28`, `v3_mnist_digit_muon_28`, and
`v3_mnist_router_ranger_28` each went from an empty output directory to
complete; `v3_mnist_router_ranger_32` went from never-run to complete.
Two directories are now mid-run that previously didn't exist:
`v3_mnist_digit_soap_32` and `v3_mnist_router_ranger_64` (each has
`_best.pt` + `_resume.pt` + one CLI transcript, no final export yet).
`v3_mnist_digit_soap_28` and `v3_mnist_digit_adamw_28` are shown with
both CLI transcripts spelled out in full (rather than the usual
shorthand) since each now has two, evidence of a rerun. The summary
counts changed from 4 complete / 1 mid-run / 5 empty-dir / 10 never-run
to 9 complete / 2 mid-run / 1 empty-dir (`v3_mnist_digit_soap_64`,
unchanged) / 8 never-run. No other files in the tree changed.

### Why

William asked for the README to be updated to include all the files
currently in the folder and subfolders.

### Source

Direct instruction from William (this conversation). Verified against a
fresh recursive file listing (`find -type f`) and directory listing
(`find -type d`, plus `ls -la` on directories that could be empty) of
`E:\mnist_v3`, rather than assumed from the previous README text or
memory of earlier conversation turns.

---

## 2026-07-29 — Add `.gitignore` (new file)

### What changed

Created `.gitignore` at the repo root with four entries: `.claude/`,
`.vscode/`, `__pycache__/`, and `*.pt`. No other file was changed.

### Why

William is pushing this project to its existing GitHub repo
(`github.com/BECKHAMW3233/mnist-ocr-ensemble-v3`) and wants Claude
Code's own local session settings (`.claude/`), VS Code's workspace
settings (`.vscode/`), and Python bytecode cache (`__pycache__/`) kept
out of version control as tooling artifacts, not project content —
consistent with how `README.md`'s "Current on-disk contents" section
already described those three as intentionally excluded from its tree.
He also wants only the files needed to run inference and review
training results tracked from each model output folder (`.onnx`
exports, `_log.csv`, `_curves.png`, and CLI transcripts) rather than the
PyTorch checkpoint files (`_best.pt`, `_final.pt`, `_resume.pt`), hence
`*.pt`. A full recursive file listing (done earlier in this
conversation, before proposing the rule) confirmed `.pt` files only
ever appear inside model output folders in this repo, so one top-level
pattern covers all of them without scoping it per-directory.

### Source

Direct instruction from William (this conversation) — no external
source for the inclusion/exclusion choices themselves. One supporting
fact was independently verified rather than assumed: `*.pt` also
removes `v3_mnist_digit_soap_32_resume.pt` from what gets pushed, which
a `Get-ChildItem` size check (this conversation) put at 156 MB —
confirmed via GitHub's own documentation
(docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)
that GitHub hard-blocks any push containing a file over 100 MiB (and
separately warns, but allows, at 50 MiB), so that file would have
failed a normal push regardless of the folder-scoping preference above.

---

## 2026-07-29 — README: rewrite "Current on-disk contents" as "Current repo contents" (reflect `.gitignore`, not raw disk)

### What changed

Replaced `README.md`'s "Current on-disk contents" section with "Current
repo contents — what's tracked in git", reflecting what actually gets
pushed to GitHub rather than every file present on local disk. Removed
every `_best.pt` / `_final.pt` / `_resume.pt` line from the tree (all
now gitignored — see the `.gitignore` entry above); added `.gitignore`
itself to the root file listing (it is tracked); for the two mid-run
models (`v3_mnist_digit_soap_32`, `v3_mnist_router_ranger_64`) this
leaves only their CLI transcript in the tree, since their checkpoint
and resume file are the only other things they currently have, and
neither is tracked. Also corrected wording on `v3_mnist_digit_soap_28`
and `v3_mnist_digit_adamw_28`: both were labeled "(2 CLI transcripts —
rerun)" in the version proposed earlier today; William flagged that
multiple transcripts mean a run was stopped and resumed, not restarted
from scratch, and asked for verification rather than assumption before
it was written. Relabeled both to "(2 CLI transcripts — stopped and
resumed)" after confirming directly against the actual log content.

### Why

William wants the README to describe what actually ends up in the live
GitHub repo, not literally every byte on local disk, now that
`.gitignore` excludes `.pt` checkpoints and tooling folders. Separately,
he corrected the "rerun" label and required it be verified, not assumed,
before being written.

### Source

Direct instruction from William (this conversation) for the section's
scope/framing change. The "stopped and resumed" correction was verified
by directly reading the CLI transcript files, not assumed:
`v3_mnist_digit_adamw_28_cli_20260728_053717.txt` ends mid-epoch-87
(lines 4104-4106) with no error, immediately after an epoch-86
checkpoint save; `v3_mnist_digit_adamw_28_cli_20260728_220102.txt`
opens training with `[Resume] Loaded from epoch 86 (val_loss=0.2836,
patience=0/15)` / `[Resume] Continuing from epoch 87` (lines 63-64) — an
exact continuation, not a restart. `v3_mnist_digit_soap_28_cli_20260728_045546.txt`
independently shows the same pattern: `[Resume] Loaded from epoch 52
(val_loss=0.2863, patience=4/20)` / `[Resume] Continuing from epoch 53`
(lines 61-62). Both scripts' resume behavior is provided by the shared
`common/checkpointing.py` module per this README's own "Modularized"
note above.

---

## 2026-08-02 — EMNIST orientation bug fixed; EMNIST ByClass letter loader added (Part 4 prep — letter-identity models)

### What changed

`supplementary_data.py`:
1. New `_correct_emnist_orientation(img)` helper, applied inside
   `EMNISTDigitsDataset.__getitem__`, `BalancedEMNISTDataset.__getitem__`,
   and the new `EMNISTByClassDataset.__getitem__` (below), right after each
   reads its raw image from the underlying `torchvision.datasets.EMNIST`
   instance and before any transform runs.
2. New `EMNISTByClassDataset` (all 62 byclass classes), `_LetterOnlyDataset`
   (filters to one case's 26 classes, remaps to dense 0-25 labels), and
   `load_base_emnist_letters(case, ...)` (train/val/test split, mirrors
   `load_base_mnist()`/`load_base_usps()`'s own mechanics) — the data layer
   for the upcoming v3 uppercase/lowercase letter-identity models
   (`v3_mnist_letter_{uc,lc}_{optimizer}_{res}.py`, added in the training-
   script phase of this same session).
3. Module docstring updated (4 additions instead of 2; new "EMNIST ByClass"
   entry in the LETTER DATASETS list).

### Why

**Orientation fix:** torchvision's `EMNIST` dataset class returns images
rotated 90° and mirrored relative to upright, for every split — a real
domain mismatch against real-world scanned character crops (which are
upright) for any model trained on EMNIST-Digits or EMNIST-Balanced data.
Found while designing the new byclass loader, not something William asked
me to go looking for. He confirmed: fix it everywhere now (not just in the
new loader), and he'll separately decide when to retrain the models that
used the uncorrected data.

**EMNIST ByClass over EMNIST Balanced for the letter models:** Balanced only
has 11 distinct lowercase classes (the other 15 are merged into their
uppercase counterpart by the dataset's own creators); William wants full
26-class uppercase and lowercase models, so ByClass (all 26 real per case)
is the only source that provides that.

### Source

- **Orientation bug:** confirmed by reading the actually-installed
  `torchvision==0.28.0` source directly
  (`site-packages/torchvision/datasets/mnist.py`) — `EMNIST.__getitem__` is
  inherited unchanged from `MNIST.__getitem__`, which does
  `Image.fromarray(img.numpy())` with no rotation/flip applied, for every
  split. Corroborated by
  [pytorch/vision#8783](https://github.com/pytorch/vision/issues/8783)
  (open, reported Dec 2024): images come out rotated 90° and mirrored; the
  documented community fix is `rotate(img, -90)` then a horizontal flip,
  which is exactly what `_correct_emnist_orientation()` does. Not
  independently verified by rendering an actual image from this project's
  own dataset (that requires running code, which is William's to run, not
  mine) — flagged to him as the one thing I couldn't confirm myself; he
  approved proceeding on the source-read + GitHub-issue evidence.
- **ByClass label ordering matches this project's own 0-9/10-35/36-61
  convention with no remapping needed:** also confirmed by reading the same
  torchvision source — `EMNIST.classes_split_dict["byclass"] =
  sorted(string.digits + string.ascii_letters)`, which sorts by ASCII value
  into exactly digits-then-uppercase-then-lowercase.
- **EMNIST Balanced's 11-vs-26 lowercase-class gap:** already documented in
  this file's own 2026-07-27 entry ("Bugs found while re-enabling EMNIST
  Balanced") and in `BALANCED_TO_BYCLASS`'s comment in `supplementary_data.py`.
- Everything else (which files to touch, scope of the fix, naming/design of
  the new loader) is direct instruction from William (this conversation).

### Verification

`python -m py_compile supplementary_data.py` — clean. No training run, no
dataset download, no rendered image — all of that is out of scope for this
change per the project's Testing rule; William owns running/verifying it.

---

## 2026-08-02 — Uppercase letter-identity canonical templates added (SOAP/AdamW/Muon @ 28x28)

### What changed

Three new training scripts, the canonical templates for the full 24-script
uppercase/lowercase letter-identity model set:
- `v3_mnist_letter_uc_soap_28.py`
- `v3_mnist_letter_uc_adamw_28.py`
- `v3_mnist_letter_uc_muon_28.py`

Each is its corresponding digit-ensemble script
(`v3_mnist_digit_{soap,adamw,muon}_28.py`) with: `NUM_CLASSES` 10 → 26,
`LABEL_MAP` → `list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")`, data loading replaced
with a `load_letters()` wrapper around the new
`supplementary_data.load_base_emnist_letters(case="upper", ...)` (added
in the previous entry), and a `HAS_SUPPLEMENTARY`-gated early exit in
`run_training()` (this data source is required, not optional, unlike the
digit ensemble's supplementary sources) — mirroring the same guard
`v3_mnist_router_ranger_*.py` already uses for the same reason. Every
optimizer-specific mechanic (SOAP's Kronecker preconditioning, Schedule-
Free AdamW's train()/eval() toggle + BatchNorm warmup, Muon's param-group
split) is unchanged from the corresponding digit script.

**One real deviation, not a straight copy:** `v3_mnist_letter_uc_adamw_28.py`
does NOT reuse the digit AdamW script's `get_class_weights()`/
`supplementary_data._extract_targets()` for its `WeightedRandomSampler`
weights. Traced through `_extract_targets()`'s `isinstance`/`hasattr`
branches by hand: it doesn't recognize the new `EMNISTByClassDataset`/
`_LetterOnlyDataset` wrapper types, and a `Subset` wrapping a
`_LetterOnlyDataset` falls through every branch to
`raise ValueError(f"Cannot extract targets from dataset type: {type(dataset)}")`
— reusing it as-is would crash. Fixed by computing sample weights directly
from the `train_targets` tensor `load_base_emnist_letters()` already
returns (`bincount` → inverse-frequency → index), which is exactly the
pattern `v3_mnist_digit_soap_28.py`/`v3_mnist_digit_muon_28.py` already use
for their own no-supplementary-data fallback — not a new invention, an
existing in-project pattern applied where the AdamW script's own approach
doesn't fit. Documented in the file's own module docstring.

### Why

Direct instruction from William: build the letter-identity models "based
on my current digits models" — same architectures, same hyperparameters,
same infrastructure (`common/`), retargeted at 26-class letter data. SOAP
was built first as the most direct template (simplest data-loading
adaptation — its digit script already had the no-supplementary-data
inline-weighting fallback this reuses almost verbatim); AdamW and Muon
followed as siblings at the same resolution before generating the
remaining 21 resolution/case variants (a separate, later step).

### Source

- Architecture/hyperparameter values: read directly from
  `v3_mnist_digit_soap_28.py`, `v3_mnist_digit_adamw_28.py`,
  `v3_mnist_digit_muon_28.py` in full (not assumed from the 64x64 variants
  or from memory) — confirmed byte-for-byte which lines are resolution-
  specific (only `IMG_SIZE` + docstring, per the diff work in the previous
  restructure's own resolution-ladder section) before writing anything.
- `_extract_targets()` crash risk: confirmed by reading
  `supplementary_data.py`'s actual `_extract_targets()` implementation and
  tracing the `Subset(_LetterOnlyDataset)` case through its branches by
  hand, not assumed.
- `HAS_SUPPLEMENTARY`-gated exit pattern: copied from
  `v3_mnist_router_ranger_*.py`'s own `run_training()`, which already
  faces the same "this data source is required, not optional" situation.
- File-touch scope, naming (`v3_mnist_letter_uc_*`), and the "no digit
  mixing" constraint: direct instruction from William (this conversation).

### Verification

`python -m py_compile` on all three files — clean. No training run — out
of scope per the project's Testing rule; William owns running it. Real
EMNIST ByClass class-imbalance severity, actual convergence, and per-class
accuracy are all unverified and untested — this entry covers only that
the three scripts are structurally sound and load-bearing logic (the
weight-computation fix above) is correct by inspection, not that training
them produces a good model.

---

## 2026-08-02 — Remaining 21 letter-identity scripts generated (full 24-script set complete)

### What changed

Generated the rest of the 24-script uppercase/lowercase letter-identity
set from the 3 approved canonical templates
(`v3_mnist_letter_uc_{soap,adamw,muon}_28.py`, previous entry):
- Resolution variants: `v3_mnist_letter_uc_{soap,adamw,muon}_{32,64,128}.py` (9 files)
- Lowercase variants at all 4 resolutions: `v3_mnist_letter_lc_{soap,adamw,muon}_{28,32,64,128}.py` (12 files)

Full 24-script set: `v3_mnist_letter_{uc,lc}_{soap,adamw,muon}_{28,32,64,128}.py`.

**How they were generated, not just what exists:** written by a small,
purpose-built Python script (not by hand, not by a blind find-and-replace)
that applies a fixed list of exact, multi-word anchor strings per
substitution point (filename references, docstring resolution/case
wording, `IMG_SIZE`, `LETTER_CASE`, `LABEL_MAP`, output paths, print
banners) — each anchor is asserted to match the source text exactly the
expected number of times (1 for single-occurrence anchors, all
occurrences for the `_uc_`→`_lc_` filename substitution) before being
applied, raising instead of silently mismatching. This was deliberately
NOT a bare-number substitution (e.g. blindly replacing every `"28"` with
`"64"`) — that would have corrupted architecture constants that happen to
contain the same digits (e.g. the classifier head's `128`-unit hidden
layer, SOAP's `1024`/`256`/`128` layer widths) and citation lines that
must NOT track resolution (the "(28/32/64/128; there is no 16x16 letter
tier...)" ladder-list line, which lists all four tiers and is identical
in every file).

### Why

Direct instruction from William to build all 24 models; per the approved
plan, the 21 non-canonical files are pure mechanical substitution of the
already-approved templates (same architecture, same hyperparameters, same
infrastructure — only resolution/case differ), so generating them
programmatically with hard verification is safer and more accurate than
retyping ~700-900 lines by hand 21 times.

### Source

- The substitution-point list itself was derived by direct inspection of
  each of the 3 template files (grep for every literal `"28"` occurrence,
  read in context, classified by hand as resolution-specific vs.
  architecture-constant vs. ladder-list-that-must-not-change) — not
  assumed from the earlier digit-ensemble resolution-diff finding, though
  it's consistent with it.
- Direct instruction from William for scope (all 24) and to proceed with
  the generated batch after review.

### Verification

- `python -m py_compile` on all 21 generated files, individually and
  after copying into the project folder alongside the 3 templates (24
  total) — clean.
- Every one of the 21 generated files has an **identical line count** to
  its source template (checked programmatically across all 21, not
  sampled) — confirms every substitution was a same-line text replacement,
  never a line insertion/deletion that could silently shift or duplicate
  code.
- Representative `diff` against the source template shown and reviewed
  for: a resolution-only variant (`uc_soap_64`), a case-only variant
  (`lc_soap_28`), and two combined resolution+case variants
  (`lc_muon_128`, `lc_adamw_64`) — each diff contained only the intended
  substitution lines, nothing else.
- **Bug caught and fixed during generation, before any file reached the
  project folder:** the generator's first run wrote files with CRLF line
  endings (Python's default text-mode write behavior on Windows), while
  every existing file in this project uses LF — caught by running `file`
  on a generated output and comparing against an existing project file,
  fixed by passing `newline="\n"` to both the template read and the
  generated-file write, then regenerated and reverified (compile +
  line-count + diff checks above are all from the corrected run).
- Real EMNIST ByClass class-imbalance severity, actual convergence, and
  per-class accuracy remain unverified and untested for all 24
  scripts — training is William's to run.

---

## 2026-08-02 — README.md updated for the 24 letter-identity models

### What changed

`README.md`: header/Overview paragraphs mention the 24 letter-identity
models; new bullet in "What changed in v3" describing them and the EMNIST
orientation fix; the projected "Repository structure" tree and the actual
"Current repo contents" tree both gain all 24 new files (and, in the
projected tree, their eventual output directories) following the
existing per-script enumeration convention; `External data locations`'s
`DATA_DIR` bullet now lists EMNIST ByClass alongside the sources already
there; `Usage` gains a "Train one letter-identity model" example. Also
fixed a wording inconsistency this same change introduced: the
"Router added" bullet used to say the router was built "ahead of a
planned, not-yet-built letter-reading phase" — now reads "ahead of the
letter-identity models (see below)", since that phase now exists.

### Why

Per CLAUDE.md's README-maintenance rule: an approved change that adds
files the README describes gets the README updated in the same change,
not left stale.

### Source

- Actual current file/directory listing confirmed via `ls`/`git status`
  in this session (not assumed from memory) before writing the tree
  additions — confirmed all 24 `.py` files exist, no output directories
  exist yet for any of them (none have been trained), and none of the 24
  are yet committed to git (all show `??` in `git status --short`).
- Content/wording: direct instruction from William, including the
  separate confirmation to fix the now-stale "not-yet-built" wording
  found while reviewing this same change.

### Verification

Read back the edited sections after writing (`grep -n "letter" README.md`
across the whole file) to confirm every mention is internally consistent
and no stray old wording was left behind.

---

## 2026-08-02 — Letter-identity ensembles wired into ocr_pipeline_mnist.py

### What changed

`ocr_pipeline_mnist.py`, 12 changes total:
1. New `UC_LABELS`/`LC_LABELS` constants (dense 0-25 A-Z / a-z — must
   match each letter script's own `LABEL_MAP` order, a cross-file
   contract same as the existing `ROUTER_LABELS` one).
2. `predict_char_topn()` gained a `labels=None` param (defaults to the
   existing `LABELS`), so it can run against a letter ensemble too —
   zero behavior change for existing digit-ensemble call sites (none of
   which pass `labels`).
3. `short_model_name()` gained a `v3_mnist_letter_(uc|lc)_{optimizer}_{res}`
   regex arm, matching the existing `digit_`/`router_` arms' style.
4. `run_pipeline()` gained `uc_sessions/uc_img_sizes/uc_model_names,
   lc_sessions/lc_img_sizes/lc_model_names` params (all default
   `None`/`[]` — omitting them reproduces the exact original behavior).
5. The `elif router_verdict in ("UC", "LC")` branch: if a matching
   letter ensemble was loaded, runs it (single-model shortcut or
   `vote_topn()`, same pattern the digit ensemble already uses) so the
   box resolves to an actual letter instead of a bare `"UC"`/`"LC"` tag.
   `agreement` deliberately stays the literal `"ROUTER-UC"`/`"ROUTER-LC"`
   string regardless of the ensemble's vote outcome — see "Why" below for
   the reasoning, this was not an oversight.
6. CHARACTER DETAIL: shows the letter ensemble's own per-model top-3
   breakdown (via a new `letter_detail` tuple field, see below) when one
   was loaded; falls back to the original "digit ensemble skipped"
   message when it wasn't.
7. New `_load_onnx_model_list()` helper, extracted from the digit
   ensemble's existing ~65-line inline CUDA-first/CPU-fallback loading
   loop in `__main__` — reused for the new `--letter-uc-models`/
   `--letter-lc-models` loading instead of a third near-identical copy.
   The digit ensemble's own call site now calls this helper too (pure
   extraction — same logic, same behavior, verified by inspection, not
   rewritten).
8. New `_is_router_or_letter_onnx()` helper + fix: `--model-dir`'s scan
   now excludes router/letter `.onnx` filenames from the digit ensemble
   — see "Bug found" below.
9. New CLI flags: `--letter-uc-models`/`--letter-uc-model-dir`,
   `--letter-lc-models`/`--letter-lc-model-dir` (each an optional
   mutually-exclusive pair, mirroring `--models`/`--model-dir`'s shape).
   Directory scans are filtered to `v3_mnist_letter_{uc,lc}_*.onnx`
   specifically (same discipline as item 8).
10. New loading blocks in `__main__` for the UC/LC ensembles, using the
    new helper, placed after the router-loading block.
11. The final per-image `run_pipeline()` call passes the new session
    lists through.
12. One new usage example in the argparse epilog.

**Structural note — a new `letter_detail` tuple field, not a repurposed
one:** the per-box `ensemble_line` tuple (`winner, agreement, raw_top1s,
all_top3, router_verdict, router_prob`) had its `raw_top1s`/`all_top3`
shaped to the DIGIT ensemble's model count — they feed the digit-indexed
INDIVIDUAL MODEL PREDICTIONS and PER-MODEL SUMMARY sections. A
letter-ensemble result has a different length/model identity, so it
could NOT safely overwrite those without risking an index mismatch in
those sections. Added a 7th tuple field, `letter_detail` (`None`
normally, or `(letter_model_names, letter_top3)` for a letter-ensemble-
backed box), read only by CHARACTER DETAIL — every other section is
completely untouched by this addition.

### Why

**`agreement` stays `"ROUTER-UC"`/`"ROUTER-LC"` literally, even when an
ensemble backs the box (not `"ROUTER-UC-MAJORITY"` etc.):** traced every
place `agreement` is branched on in this file (ENSEMBLE RESULT
rendering, PAGE LAYOUT rendering, CHARACTER DETAIL's flag/message logic,
PER-MODEL SUMMARY's per-digit-model bracket tags, and — critically — the
`_excluded` tuple used by both PER-MODEL ACCURACY and ACCURACY scoring to
exclude letter boxes from `--ground-truth` digit-accuracy comparisons).
Introducing new agreement values for letter-ensemble outcomes would have
required touching all of those, and would have silently stopped
excluding letter predictions from digit accuracy scoring (since
`_excluded` checks the literal string `"ROUTER-UC"`/`"ROUTER-LC"`) unless
every one of those checks was also updated in lockstep — a much larger,
higher-risk change for no real benefit, since `winner` already carries
the actual letter, and CHARACTER DETAIL already carries the router's own
verdict/confidence separately via `router_verdict`/`router_prob`.

**Extracting `_load_onnx_model_list()` instead of writing a third
copy:** the digit ensemble's loading loop (CUDA-first, per-model CPU
fallback, the "CUDA accepted but actually running on CPU" cuDNN-mismatch
diagnostic) is ~65 lines; the new UC/LC loading needs the identical
mechanics twice more. Copy-pasting it two more times would have meant
three places to keep in sync for any future fix to that logic.

### Bug found (pre-existing, unrelated to this task, fixed as part of
### making letter-model loading safe)

`--model-dir`'s recursive scan collected every `.onnx` file with no name
filtering at all — meaning a plain `--model-dir .` run would already
silently load `v3_mnist_router_ranger_*.onnx` into the *digit* ensemble
(10-class softmax assumed) even before this session's changes, and would
have started doing the same to the 24 new letter `.onnx` files the
moment they're trained. Found while designing where the new
`--letter-uc-model-dir`/`--letter-lc-model-dir` flags should scan and
realizing the same gap would apply to them too. Fixed for all three
(`--model-dir`, `--letter-uc-model-dir`, `--letter-lc-model-dir`) in the
same change, per William's direction to also do the pipeline wiring
properly rather than leave a known new gap.

### Source

- Every touched section (`predict_char_topn`, `short_model_name`,
  `run_pipeline`, the per-box dispatch loop, CHARACTER DETAIL, the
  `--model-dir` scan, the digit-ensemble loading block, `__main__`'s
  argparse/loading structure) was read directly and in full before being
  edited — not assumed from an earlier subagent summary (that summary
  was used only to scope which sections mattered, not as the basis for
  the actual edits).
- The `_excluded`-tuple/accuracy-scoring dependency on the literal
  `"ROUTER-UC"`/`"ROUTER-LC"` strings was confirmed by grepping every
  `agreement ==` / `agreement in (...)` comparison in the file and
  reading each one in context, not assumed.
- Scope and design decisions (naming, CLI flag shape, fixing the
  `--model-dir` gap now rather than deferring it): direct instruction
  from William, per the plan approved earlier this session.

### Verification

`python -m py_compile ocr_pipeline_mnist.py` — clean. `ast.parse()` used
to confirm all 5 new/modified top-level functions exist with the
expected names. Grepped every `ensemble_line.append(...)` call site and
the CHARACTER DETAIL unpack line to confirm all three now consistently
construct/consume the same 7-element tuple shape (no stragglers left at
the old 6-element shape, which would have raised `ValueError` on
unpacking). Grepped every `uc_sessions`/`lc_sessions`/`uc_img_sizes`/
`lc_img_sizes`/`uc_model_names`/`lc_model_names` reference to confirm
each is defined in `__main__` before the final `run_pipeline()` call
that passes them through, and defined as a parameter before use inside
`run_pipeline()` itself. **Not verified:** no actual image was run
through this pipeline (no trained letter `.onnx` files exist yet — all
24 letter scripts are untrained, see the earlier 2026-08-02 entries) — so
the letter-ensemble dispatch path, the CHARACTER DETAIL letter-breakdown
rendering, and the new CLI flags have only been checked by static
inspection, not by an actual run. That's William's to verify once at
least one letter model is trained.

### README.md updated for this same change

The "Run the inference pipeline" usage example: fixed the `--model-dir`
comment (previously said "every .onnx found recursively", no longer
accurate now that router/letter files are excluded — see the "Bug found"
section above) and added a fourth example showing `--letter-uc-model-dir`/
`--letter-lc-model-dir` combined with `--router-model`. Per CLAUDE.md's
README-maintenance rule — this documents a real CLI behavior change from
the same edit, not a separate task.

---

## 2026-08-02 — run_all_training.ps1 updated for the 24 letter-identity scripts (24-script letter model set now complete end-to-end)

### What changed

`run_all_training.ps1`: header comment updated (20 -> 44 scripts,
describes the new letter-identity block); `$scripts` array gains a new
block of 24 entries after the existing 20 (one row per resolution —
28/32/64/128 — each listing uc-soap/uc-adamw/uc-muon/lc-soap/lc-adamw/
lc-muon, matching the existing digit block's one-row-per-resolution
layout); completion message updated from "All 20 scripts complete." to
"All 44 scripts complete." `$root`/`$python` (both pre-existing,
machine-specific hardcoded paths) were left untouched, per this project's
own "don't fix system-specific paths unless asked" convention.

This closes out the full letter-identity-models task: all 24 scripts
exist (earlier 2026-08-02 entries), are wired into the inference pipeline
(previous entry), and are now included in the one-command full-training-
run orchestration.

### Why

Direct instruction from William to add all 24 scripts to the run-all
list, completing the same task the earlier phases (canonical templates,
generated variants, pipeline wiring) were building toward.

### Source

Current file read in full before editing (not assumed from memory) to
get the exact existing array formatting/style to match. Script list and
ordering: derived from the same file-name set already verified present
on disk in the "21 letter-identity scripts generated" entry earlier in
this file, not retyped from scratch — cross-checked via `grep -c` after
writing that the array contains exactly 44 unique `"v3_mnist_*.py"`
entries with zero duplicates.

### Verification

- `grep -o '"v3_mnist[^"]*\.py"' run_all_training.ps1 | wc -l` → 44;
  `| sort | uniq -d` → empty (no duplicates).
- `[System.Management.Automation.Language.Parser]::ParseFile(...)` (pure
  syntax parse, no execution) → no errors.
- **Not verified:** the script was not actually run (would mean training
  models — William's to run, per the project's Testing rule), so the
  skip-if-`_final.pt`-exists logic and the actual training sequence for
  the new entries are unverified beyond static parsing and the file-list
  cross-check above.

---

## 2026-08-02 — Changelog placement bug found and fixed (this entry and the 5 before it were misfiled mid-document, now corrected)

### What happened

While adding this entry, discovered that all 5 of today's earlier
changelog entries (EMNIST orientation fix, canonical templates, 21
generated scripts, README update, and ocr_pipeline_mnist.py wiring) had
been inserted in the *middle* of this file instead of appended at the
true end. Root cause: my very first edit of this session anchored on
text I remembered from an initial read of this file that had been
silently truncated by the harness at line 983 of what was then a
2321-line file — I mistook that truncation point for the file's actual
end and anchored my first insertion there, which happened to land
between two unrelated 2026-07-27/07-28 subsections of the same
top-level restructure entry. Every entry after that one chained
correctly onto the one before it, so the 5 entries were internally
consistent and complete — just collectively in the wrong place, splitting
"Minimum steps/epoch floor added"'s subsection away from the "DataLoader
worker count auto-scaled" subsection that originally followed it
directly.

**Nothing was lost, deleted, or altered — William's direct instruction,
given mid-session, was to append only and never lose any part of the
record.** Fixed by mechanically relocating the exact same 422 lines
(verified via a line-multiset equality check, not just visual
inspection) from their mid-file position to the true end of the file,
restoring the original adjacency between the two 2026-07-27/07-28
subsections they'd been wedged between. No content was rewritten, only
repositioned.

### Why

Direct instruction from William, given explicit choice between leaving
the misplacement in place with a pointer note versus relocating the
entries — he chose relocation.

### Source

Verified via `grep -n`/`sed -n` against the actual current file (not
assumed) to find the exact line boundaries of the misplaced block and
confirm what came immediately before/after it on both sides. The
relocation itself was done by a small Python script, dry-run first
against a scratch copy (diffed and visually confirmed correct at both
boundaries before touching the real file), then applied — see the
script's own line-multiset equality assertion for the mechanical
correctness check.

### Verification

Dry run: line count unchanged (2743 -> 2743), sorted multiset of every
line in the file identical before and after (proves no line's content
was added, removed, or altered — only reordered). Applied to the real
file, then re-verified the same two checks against it directly, plus a
visual read of both boundary regions (`sed -n` around the old insertion
point and `tail` of the new true end) confirming the historical
subsection adjacency was restored and all 5 (now 6, including this
entry) 2026-08-02 entries read in correct chronological order at the
end of the file.

---

## 2026-08-02 — Session work delivered from the git worktree into this checkout

### What changed

Copied five files (overwriting the versions already in this checkout)
from the `claude/emnist-mnist-case-models-50a96a` git worktree, plus all
24 new letter-identity scripts: `supplementary_data.py`,
`ocr_pipeline_mnist.py`, `run_all_training.ps1`, `README.md`,
`v3_CHANGELOG.md` (overwritten), and
`v3_mnist_letter_{uc,lc}_{soap,adamw,muon}_{28,32,64,128}.py` (24 new
files). Every copied file was diffed against its worktree source
immediately after copying and confirmed byte-identical.

### Why

All of this session's actual work (the EMNIST orientation fix, the 24
letter-identity models, the `ocr_pipeline_mnist.py` wiring, the
`run_all_training.ps1` update — see every entry above this one) happened
in a git worktree, which is a separate checkout from this one. Per this
project's own worktree rule, a git merge from the worktree branch into
this checkout is never permitted under any circumstance — so getting the
work here meant copying the files directly instead, per William's direct
request ("put in my folder on my pc").

### Source

Direct instruction from William. File list and copy method: my own plan
for this request, approved by him before execution.

### Verification

`diff` of each copied file against its worktree source, immediately
after copying — all five reported identical; file count check confirmed
24 letter scripts present.

---

## 2026-08-02 — Project reorganized into model-type folders (digit_models/, router_models/, uppercase_models/, lowercase_models/)

### What changed

Created four new top-level folders and moved every training script (plus
its existing output directory, where one existed) into the matching
one:
- `digit_models/` — 15 scripts (`v3_mnist_digit_{soap,adamw,muon}_{16,28,32,64,128}.py`)
- `router_models/` — 5 scripts (`v3_mnist_router_ranger_{16,28,32,64,128}.py`)
- `uppercase_models/` — 12 scripts (`v3_mnist_letter_uc_{soap,adamw,muon}_{28,32,64,128}.py`)
- `lowercase_models/` — 12 scripts (`v3_mnist_letter_lc_{soap,adamw,muon}_{28,32,64,128}.py`)

`common/`, `supplementary_data.py`, `ocr_pipeline_mnist.py`,
`run_all_training.ps1`, `README.md`, and `v3_CHANGELOG.md` stayed at the
project root, per William's direct instruction.

**Real code change, not just a file move:** every one of the 44 moved
scripts does `from common.seeding import ...` / `from supplementary_data
import ...` as a top-level-module import, which only resolved before
because the script and `common/`/`supplementary_data.py` were siblings
at the project root (Python puts a script's own directory on
`sys.path[0]`). Moving a script one level deeper breaks that import
unless fixed. Fixed by inserting one line into all 44 scripts, at the
identical anchor point every family (digit/router/letter, every
optimizer, every resolution) shares:
```python
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
```
placed between the existing `from datetime import datetime` line and
`from common.seeding import (`. Applied via a small anchor-verified
Python script (each anchor asserted to match exactly once per file
before being touched, same method used earlier this session for the 21
generated letter scripts) — dry-run against a scratch copy first
(compiled clean, visually confirmed correct), then applied for real.

### Why

Direct instruction from William, following up on the letter-identity
models work: "I need this all put in my folder on my pc bt i need to
also have the folder reorganized into folders for mnist digit models,
upper case models, lower case models and then also router modles."

### Bug found during this change — real trained content lost, cause not conclusively identified

**What happened:** before this reorganization, several digit and router
models had real trained output (`.onnx`, `.csv`, `.png`, CLI transcripts
— the git-tracked artifacts; `.pt` checkpoints were always gitignored)
— confirmed directly (`du -sh`) before touching anything: `v3_mnist_digit_soap_64/`
was 88MB, `v3_mnist_digit_adamw_64/` was 112MB, total project ~1.6GB.
After the reorganization (folder creation, script moves, the 44-script
import-path fix, and a lightweight `--help`-based import-resolution
check on one script per folder — see "Verification" below), **every one
of the 15 digit/router output directories was found completely empty**
— confirmed both by direct `ls`/`find` across the whole checkout and by
`git status` showing the old tracked artifact paths as deleted with no
corresponding new-location entries anywhere.

**What was checked and ruled out as the cause**, not assumed:
- The import-path-fix script: its file loop is `folder_path.glob("*.py")`
  — it only ever opened/wrote `.py` files, never touched a directory.
- The `--help`-based import verification: proven not to be the cause by
  checking whether it created a new output directory for a
  never-before-trained letter model (`uppercase_models/v3_mnist_letter_uc_soap_28/`)
  — it did not, and the captured output shows the process exited cleanly
  at `argparse`'s usage text, before reaching the `OUTPUT_ROOT.mkdir()`
  line later in the same `if __name__ == "__main__":` block that every
  script family shares. Since this code path is identical across every
  script family, this rules the check out for the digit/router scripts
  too, not just the letter one actually tested.
- A concurrently running training process: `tasklist` found no running
  `python.exe` process at the time of investigation.
- Any git operation that touches the working tree: only read-only
  `git status`/`git show` were run against this checkout before the loss
  was discovered.
- `.git`'s own object database size (195MB) was checked specifically to
  rule out "the 1.6GB figure was mostly git-internal all along" as an
  alternative explanation for why total size didn't obviously drop —
  195MB is far too small to account for 1.6GB, meaning real working-tree
  content did exist post-move (consistent with the loss happening later
  in the sequence, not during the move itself) — though this places a
  bound on *when*, not a definitive cause.

**What was NOT determined:** the exact command or mechanism responsible.
This is stated plainly rather than guessed at, per this project's own
verification standard — a specific cause was actively searched for and
not found, not left unconsidered.

**Resolution, per William's direct instruction:** no recovery attempted
(git history could potentially restore the non-`.pt` tracked artifacts
via `git restore`, but that's a state-changing git operation not run
without explicit sign-off, and none was given). William confirmed
retraining every model was already the plan regardless of this finding.
`README.md`'s "Current repo contents" section was rewritten to describe
the actual current state honestly (every output directory exists but is
empty) rather than the pre-reorganization "9 complete / 2 in-progress"
status, which is no longer true.

### Source

Every claim above is something I directly checked in this session (`du`,
`ls`, `find`, `git status`, `git show --stat`, `tasklist`), not inferred
or assumed — see "What was checked and ruled out" for the specific
commands each conclusion rests on. Scope and resolution: direct
instruction from William, given after I stopped and reported the finding
rather than continuing silently.

### Verification

- All 44 moved scripts: `python -m py_compile` clean from their new
  locations.
- Real runtime verification (not just syntax): `python <script> --help`
  run for one representative script per folder (`digit_models/v3_mnist_digit_soap_16.py`,
  `router_models/v3_mnist_router_ranger_16.py`,
  `uppercase_models/v3_mnist_letter_uc_soap_28.py`,
  `lowercase_models/v3_mnist_letter_lc_muon_28.py`) — all four reached
  `argparse`'s usage output cleanly, proving the `sys.path` fix actually
  resolves `common`/`supplementary_data` imports at runtime, not just
  that the files parse.
- Directory listing confirmed exactly 4 new folders, correct file counts
  in each (15/5/12/12), and zero stray `v3_mnist_*` files/directories
  left at the project root.
- The data-loss finding itself: see the dedicated section above for what
  was and wasn't verified regarding cause.

---

## 2026-08-02 — run_all_training.ps1 reworked for the new folder layout

### What changed

Two real logic changes, not just data:
1. Every `$scripts` array entry gained its subfolder prefix (e.g.
   `"digit_models\v3_mnist_digit_soap_16.py"`).
2. The skip-check's checkpoint-path computation previously assumed the
   output directory sat directly under `$root`
   (`Join-Path $root "$baseName\${baseName}_final.pt"`) — with
   subfolder-qualified `$script` values this would have doubled the
   subfolder into the path. Fixed by deriving the checkpoint path from
   the script's own directory instead:
   ```powershell
   $scriptPath = Join-Path $root $script
   $scriptDir  = Split-Path $scriptPath -Parent
   $leafName   = [System.IO.Path]::GetFileNameWithoutExtension($script)
   $finalCkpt  = Join-Path $scriptDir "$leafName\${leafName}_final.pt"
   ```
`$root` (`"E:\mnist_v3"`) and `$python` were left untouched — both
already correct, per this project's own "don't fix system-specific paths
unless asked" convention. Header comment updated to describe the new
layout.

### Why

Direct instruction from William (the folder reorganization request), and
a necessary consequence of it — the old path-construction logic would
have silently pointed at the wrong checkpoint path for every script once
`$scripts` became subfolder-qualified, breaking the skip-already-complete
logic (not a cosmetic issue — it would have caused already-trained
models to be silently retrained, or worse, incorrectly considered
complete).

### Source

Verified the fix is correct against a real file, not just by inspection:
computed `$finalCkpt` by hand for `digit_models\v3_mnist_digit_soap_16.py`
and confirmed the resulting path
(`E:\mnist_v3\digit_models\v3_mnist_digit_soap_16\v3_mnist_digit_soap_16_final.pt`)
matched a real file that existed on disk at the time of that check (see
the "Bug found" entry above for what later happened to that file — the
path-computation logic itself was verified correct against real
evidence, independent of that later data loss).

### Verification

`[System.Management.Automation.Language.Parser]::ParseFile(...)` (pure
syntax parse, no execution) — clean, run twice (once after each edit to
this file).

---

## 2026-08-02 — README.md updated for the model-type folder reorganization

### What changed

"Repository structure" section rewritten: both the projected-full-layout
tree and the "Current repo contents" tree now nest every model script
under its `digit_models/`/`router_models/`/`uppercase_models/`/
`lowercase_models/` folder instead of flat at the root. `Usage` section's
example commands all gained their subfolder prefix. The "Current repo
contents" section's status description was rewritten from the
pre-reorganization "9 complete / 2 in-progress / ..." breakdown to state
plainly that every output directory currently exists but is empty (see
the "Bug found" entry above), rather than leaving stale completion
claims in place.

### Why

Per CLAUDE.md's README-maintenance rule — an approved change that moves
files the README describes gets the README updated in the same change.
Describing the current state honestly (empty directories, not the old
completion status) rather than preserving now-false claims follows the
same rule's spirit and this project's own verification standard.

### Source

Actual current folder contents confirmed via direct `ls`/`find` in this
checkout before writing (not assumed from the worktree's README, which
described a different, now-stale state). Wording and scope: direct
instruction from William.

### Verification

`grep` across the full README for any remaining bare (non-subfolder-
prefixed) `v3_mnist_*` script path reference in a command example —
found and fixed all instances; final pass confirmed zero remaining.

---

## 2026-08-02 — Batch-size candidate ladder extended to a 300GB-VRAM-per-device ceiling

### What changed

`common/batch_sizing.py`: `determine_batch_size()`'s default `candidates`
tuple replaced (previously `(64, 128, 256, 512, 1024, 2048, 4096, 8192,
16384)`, 9 values) with a 205-value ladder: doubles 64 -> 2048 (6
candidates), then steps by a constant +1536 up to 306176, capped with
one final point at 307200 (= 300 x 1024). Module docstring updated to
describe the new ladder's shape and rationale in place of the old "64 ->
16384" description. The binary-search refinement algorithm itself
(narrowing to within 8 of the true OOM ceiling once a working/failing
bracket is found) is completely unchanged — this was purely a
candidate-list swap.

### Why

Direct instruction from William: future-proof the batch-size search for
hardware beyond his current RTX 4080 (16GB) — a better single GPU, a
cloud instance, or a compute-cluster node with more VRAM per device —
without needing this file edited again later. Chose 300GB specifically
per his direct instruction after I researched and reported current
top-of-market per-device VRAM (see Source below); a long candidate list
costs nothing at runtime since `determine_batch_size()`'s coarse pass
stops at the FIRST candidate that fails, so untested tail candidates are
free.

### Source

Current top-of-market single-GPU/accelerator VRAM, checked via web
search (not assumed from training data, which could be stale for
fast-moving hardware releases) before proposing a ceiling:
- NVIDIA H100: 80GB HBM3 (current cloud workhorse)
- NVIDIA H200: 141GB HBM3e — live on AWS/Azure/GCP/Oracle by early 2026
  ([H200 Cloud Pricing guide, getdeploying.com](https://getdeploying.com/gpus/nvidia-h200))
- NVIDIA B200: ~180-192GB HBM3e
  ([NVIDIA Blackwell architecture announcement, wccftech.com](https://wccftech.com/nvidia-blackwell-gpu-architecture-official-208-billion-transistors-5x-ai-performance-192-gb-hbm3e-memory/))
- NVIDIA B300 (Blackwell Ultra) and AMD MI350X/MI355X: **288GB HBM3e
  each** — currently tied for the highest per-GPU VRAM on the market
  ([B200 vs MI355X comparison, bacloud.com](https://www.bacloud.com/en/blog/203/nvidia-b200-vs-amd-instinct-mi355x-next-gen-ai-data-center-gpu-showdown.html);
  [2026 GPU selection guide, vessl.ai](https://vessl.ai/en/blog/gpu-workload-guide-en))

The exact 205-value list was generated and verified programmatically
(not hand-typed across ~200 values, which would risk a transcription
error) — see Verification below.

### Verification

- Generated the list via a small Python script (`base + [2048+1536k for
  k in 1..198] + [307200]`), asserted it was strictly increasing with no
  duplicates, and printed it for direct comparison before writing it
  into the file.
- After writing, imported `determine_batch_size` from the real file and
  compared its actual default `candidates` tuple against a freshly
  recomputed expected list in a separate check — confirmed an exact
  205-element match, ruling out any transcription error in formatting
  the list across multiple source lines.
- `python -m py_compile common/batch_sizing.py` — clean.
- Not verified: an actual training run exercising any candidate above
  the low thousands (this project has only ever run on a 16GB card) —
  out of scope per the project's Testing rule; William owns running it
  if/when bigger hardware is available.

---

## 2026-08-02 — Batch-size probe fixed for shared/heterogeneous multi-GPU nodes (every rank now probes independently; global batch size is the minimum across ranks)

### What changed

`common/distributed.py`: new `all_reduce_min(value, device)` function,
added directly after `all_reduce_sum()` — takes the minimum of `value`
across every rank and returns that same minimum to all of them; no-op
passthrough when not running distributed (identical convention to
`all_reduce_sum()`/`broadcast_int()`).

`common/batch_sizing.py`: `determine_batch_size()` no longer skips the
probe loop on non-main ranks. Previously: `if is_distributed() and not
is_main_process(): return broadcast_int(0, device)` — only rank 0 ever
actually ran `_probe_batch_size()`; every other rank received rank 0's
result via broadcast, unconditionally. Now: every rank runs the full
coarse-pass-plus-refinement loop independently against its own device
(`_probe_batch_size()` already correctly targeted `device.index` per
rank, so no change was needed there), and the final return value at all
three return sites (`candidates[0]` fallback, `last_working` no-
refinement-needed case, and the refined `lo`) is `all_reduce_min(...)`
instead of `broadcast_int(...)`. Import line updated to only pull in
`all_reduce_min` (the previous `is_distributed`, `is_main_process`,
`broadcast_int` imports are no longer referenced anywhere in this file —
confirmed via grep before removing them). `determine_batch_size()`'s own
DDP-note docstring rewritten to describe the new design and why.

`broadcast_int()` itself was left completely unchanged in
`common/distributed.py` — still correct, general-purpose code; grepped
the whole project first and confirmed nothing else calls it, but removed
it from nowhere since deleting it wasn't asked for and is out of scope
for this fix.

### Why

Direct instruction from William, following a question about what
happens on a shared compute node with multiple GPUs used concurrently.
The original rank-0-broadcast design's own docstring (`broadcast_int()`)
stated its assumption plainly: "every rank's independent probe would
land on the identical number, which is only guaranteed true if every GPU
in the job is genuinely identical." That assumption holds on a
dedicated/exclusive multi-GPU box but not on a shared/contended node,
where another tenant's job can leave one rank's GPU with meaningfully
less free VRAM than another's even on physically identical hardware —
under the old design, a rank sharing its GPU with someone else could OOM
mid-training even though rank 0's probe (on a less-contended GPU)
succeeded and got broadcast to everyone. Taking the minimum across
independently-probed ranks fixes this: every rank trains at a batch size
safe for whichever rank has the LEAST available VRAM at probe time.

### Source

`broadcast_int()`'s own pre-existing docstring (read directly, not
paraphrased from memory) for the exact wording of the assumption being
superseded. The fix pattern (per-rank independent probe + collective
MIN) is the direct, symmetric counterpart to this file's own pre-existing
`all_reduce_sum()` (same `dist.all_reduce()` call, different
`ReduceOp`) — not a new pattern introduced from outside this codebase.
Scope (leave `broadcast_int()` in place rather than deleting it): direct
instruction from William ("please don't break what is already working"),
interpreted as staying minimal/conservative rather than also cleaning up
now-unused-in-this-file code that could still be useful elsewhere.

### Verification

- `python -m py_compile` on both edited files — clean.
- Grepped `common/batch_sizing.py` for any stray remaining
  `broadcast_int`/`is_distributed`/`is_main_process` reference after the
  edit — zero found.
- Direct functional check of the specific claim in this change's own
  "Why" section — that the non-distributed path is completely
  unaffected: imported `all_reduce_min` in a fresh Python process (this
  environment has `WORLD_SIZE` unset, so `is_distributed()` is False)
  and confirmed `all_reduce_min(1234, None) == 1234` — the passthrough
  returns the input completely unchanged, matching `broadcast_int()`'s
  own no-op behavior in the same scenario byte-for-byte.
- **Not verified:** actual multi-rank behavior (the MIN reduction itself
  under a real `torchrun` job) — no multi-GPU hardware available in this
  environment, same caveat `common/distributed.py`'s own module
  docstring already states for every DDP-related function in this file,
  now including this one. William owns confirming this on real
  multi-GPU (ideally genuinely shared/contended) hardware before
  trusting it for an actual run.
