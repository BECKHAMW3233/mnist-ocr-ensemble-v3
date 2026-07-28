"""
common/
=======
Shared modules imported by every v3 training script (the 15 digit models
and the 5 router models). Pulled out of the v2 project's per-script
duplication (auto batch-sizing, checkpoint/resume, telemetry, seeding,
CLI logging, ONNX export) into one place each, per the v3 modularization
pass — see v3_CHANGELOG.md for the reasoning and file-by-file diff summary.

This is a structural refactor only: every function here is a direct
extraction of logic that was previously copy-pasted near-verbatim across
12-16 v2 scripts, with only the truly per-script bits (model/optimizer
construction, resolution, output paths) left as caller-supplied
parameters. No behavior changes were made while extracting.
"""
