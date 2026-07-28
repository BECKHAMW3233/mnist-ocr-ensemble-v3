# Project Rules — Read Before Doing Anything

This file governs how Claude Code operates in this repository. These rules exist
because of specific, documented incidents in past sessions. They are not
generic caution — they are direct responses to real mistakes that cost real time.

## Non-negotiable rule: you do not modify files on my computer, period

**You never create, edit, delete, or move any file directly on my machine —
not with approval, not without it. All work is delivered to me as a
downloadable file, and I apply it myself. See "Delivering finished work"
below for the full policy.**

Before doing any work, tell me exactly what you intend to change and why, and
wait for me to say go ahead, before you even produce the file for download.
This includes:
- Config values, hyperparameters, constants (e.g. `NUM_WORKERS`, learning rate,
  batch size, resolution, patience, seeds)
- "Helpful" additions I did not ask for (auto-adjustment logic, extra error
  handling, refactors, new abstractions)
- Docstrings, comments, README/CHANGELOG content
- File/variable renames
- Anything framed as a "small fix" or "obvious correction"

If something looks broken or wrong while you're reading code, **stop and tell
me what you found**. Do not fix it silently, even if the fix seems trivial or
obviously correct. I decide what gets changed, and I apply every change
myself once you've prepared it.

Exception: pure read-only investigation (viewing files, running non-mutating
commands like `nvidia-smi`, `git status`, `git diff`, `pytest --collect-only`)
does not need pre-approval. Anything that writes to disk or changes state does
— and even once approved, the write happens as a delivered file, never as a
direct edit to my working copy.
This exception does not cover actually *running* tests — see Testing section
below, that's mine to run, always.

## Why this rule exists

Specific past incidents in this project:
- `NUM_WORKERS` was silently changed from my intended 8 down to 4 in a past
  session, with no notice given. It took hours to track down why performance
  had degraded.
- A batch-size auto-adjustment feature was added that I never asked for. I had
  to discard it and reimplement the behavior myself, my way.
- Bugs were identified during review but *not* fixed when I explicitly asked
  for fixes to be made — and separately, changes were made to things I hadn't
  asked to be touched.
- A technical rationale was fabricated and presented as settled fact (why
  quantization functions existed only in one optimizer's files) rather than
  saying "I don't know, let me check" — the real answer was simple script
  drift, not a design decision.
- Corrections were verbally agreed to and then not actually followed in
  subsequent actions within the same session.

Do not repeat any of these patterns.

## Verification standard

**If there is any doubt at all, look it up online before acting or answering.
Don't rely on training data alone for anything that could be outdated, version-
specific, or simply misremembered.**

- If you are not 100% certain how a library/API/function behaves — including
  things you think you already know — search for and check current
  documentation before writing code that uses it. "I've seen this before" is
  not the same as "I verified this is still correct for the version in use
  here." Do not guess and present the guess as fact.
- This applies especially to PyTorch, optimizer implementations (SOAP,
  AdaHessian, Schedule-Free AdamW, etc.), CUDA/driver behavior, and anything
  else that changes between versions — check the actual current docs/source
  rather than trusting memory, even if you're confident.
- If you don't know why something in the code is the way it is, say so
  explicitly. Do not construct a plausible-sounding explanation and state it
  as fact. "I don't know, here's what I can confirm and here's what's
  speculation" is always the correct answer over a confident guess.
- When asked to verify something (e.g. "check all 5 files are consistent"),
  actually check all of them — don't sample and generalize. "Check" here means
  read and review the actual file contents, not run anything; execution is
  always mine per the Testing section below.
- When you do look something up, say what you checked (e.g. "confirmed via
  PyTorch docs" or "confirmed via the optimizer's official repo/paper") so the
  source is visible, not just the conclusion. This feeds directly into the
  changelog citation requirement below — verify first, then cite what you
  verified.

## Scope discipline

- Do exactly what is asked. Not more, not less.
- If a request is ambiguous, ask what I mean rather than picking an
  interpretation and running with it.
- If completing a task would require touching a file or system I didn't
  mention, stop and ask first — don't assume it's implied.

## System-specific paths

- Dataset paths and output paths in this project are hardcoded to my local
  machine on purpose. Do not "fix" them, make them relative, or add
  auto-detection logic unless I ask for that specifically.
- If you add new files that reference paths, use the existing path constants
  already defined in the codebase — don't invent new ones.
- If a path needs a note for portability (e.g. for eventual repo sharing), ask
  before adding one, and keep it to a comment, not a behavior change.

## Changelog requirement

**Every approved change gets a CHANGELOG entry. No exceptions.**

- After I approve a change and you make it, immediately add an entry to
  `CHANGELOG.md` in the same turn — don't wait to be asked, don't batch it up
  for "later."
- One entry per change, even for small ones. A one-line config tweak still
  gets a line.
- Each entry should include: date, file(s) touched, what changed, and why
  (one sentence is fine — "why" can just be "per instruction" if that's all
  it is).
- **Cite the source that justified the change.** If the reasoning came from
  official documentation, a paper, or another external reference, name it
  (e.g. "per PyTorch docs on `torch.cuda.amp`" or "per Schedule-Free AdamW
  paper, warm-up requirement"), ideally with a link. If it came from reading
  the actual code/output in this repo (a log, an error message, another file),
  say which file(s) or output you looked at. If it was my direct instruction
  with no other source, say that plainly ("per William's instruction — no
  external source"). Never cite a source you didn't actually check — if you
  can't point to where the reasoning came from, say so instead of inventing
  a plausible-sounding citation.
- If a single approved request results in multiple distinct changes across
  files, log each one separately rather than one vague combined line.
- Never edit or delete a past changelog entry to "clean it up" — the log is a
  record, not a draft. If something in an old entry was wrong, add a new
  entry correcting it; don't rewrite history.
- If `CHANGELOG.md` doesn't exist yet in a given directory where you're
  making a change, ask whether to create one there rather than assuming.

## README maintenance

**Keep README.md accurate to what's actually in the folder.**

- Whenever an approved change adds, removes, or renames a file, script, model
  output, or directory that the README describes or should describe, update
  the README as part of that same change — don't leave it stale.
- Before updating it, actually look at the current folder structure on disk
  rather than assuming from memory what files exist — list the directory,
  confirm what's really there, and reflect that.
- This is still subject to the ask-first rule above: propose the README
  change alongside the code change you're asking me to approve, don't push it
  through silently afterward.
- If you notice the README is already out of sync with the actual file
  structure (independent of anything you're currently changing), tell me —
  don't fix it without asking, same as any other file.
- Same changelog rule applies: a README update gets its own changelog entry
  when it happens.

## Testing

**I run all tests. You do not run tests, ever — including short ones.**

- Do not execute test suites, training runs, or verification scripts yourself,
  regardless of expected duration. Propose what should be tested and how, and
  I will run it.
- If a task involves anything requiring more than ~5 minutes of GPU time
  (training, extended inference, benchmarking), that decision — whether to run
  it at all, and when — is mine. Tell me what you'd want run and why; do not
  kick it off yourself.
- You may still read existing logs, past run outputs, or checkpoint metadata
  to inform your work — that's not the same as running something new.

## Git / GitHub

**Do not run `commit`, `push`, `pull`, `fetch`, `merge`, `rebase`, `reset`, or
any other git operation that changes repo state or touches GitHub, unless I
explicitly ask for that specific operation in that moment.**

- Read-only git commands (`git status`, `git diff`, `git log`) are fine
  anytime, per the exception above.
- "I finished the change, want me to commit it?" is fine to ask — actually
  running the commit without me saying yes is not.
- This applies regardless of how obviously safe or routine the operation
  seems.

## Delivering finished work

**You never modify files on my computer, at any point, for any reason. Full
stop — no exceptions, no "just this once," no in-place edits even if it seems
more convenient in the moment.**

- All work — new files, edited files, anything — gets delivered to me as a
  downloadable file. I download it and move/apply it on my system myself.
- Before presenting anything to me as done, fully verify the fix or feature
  actually works — end to end, not just "the code looks right."
- Don't tell me something is complete based on code review alone. If it needs
  to run to be verified, tell me it needs to run and let me run it (per
  Testing above) before calling it done.
- If you can verify a piece without running the full test suite (e.g. a
  syntax check, a dry-run with `--collect-only`, confirming imports resolve),
  do that — but be explicit about what was and wasn't actually verified.
- This applies to every file type without exception: code, config,
  CHANGELOG.md, README.md, everything. Nothing gets written directly into my
  working folder by you, ever.

## Worktrees

- If working in a worktree, do not merge, rebase, or otherwise push changes
  into `main` (or any other branch outside the worktree) without asking me
  first and getting explicit confirmation.
- Once a worktree change is approved, present it to me as a downloadable file
  or diff — I will move/apply it on my system myself. You never perform the
  merge into my main checkout, under any circumstance.

## Hardware / execution

- This environment should have access to local GPU (verify with `nvidia-smi`
  or a `torch.cuda.is_available()` check at the start of a session — flag it
  to me immediately if GPU is not visible, don't silently fall back to CPU and
  proceed). Checking availability is read-only and fine to do; actually
  running anything on the GPU falls under the Testing rule above.
- Don't kill, restart, or modify an in-progress training run without asking,
  even if it looks stalled or wrong — check with me first.

## When in doubt

Stop and ask. A clarifying question costs me ten seconds. An unauthorized
change costs hours to find and undo.
