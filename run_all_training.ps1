# run_all_training.ps1
# =====================
# Runs all 44 v3 training scripts in order: the 20 digit/router scripts
# (16x16 -> 28x28 -> 32x32 -> 64x64 -> 128x128, each resolution as soap ->
# adamw -> muon -> router), then the 24 letter-identity scripts (28x28 ->
# 32x32 -> 64x64 -> 128x128 -- no 16x16 tier, that's USPS-digit-only --
# each resolution as uc-soap -> uc-adamw -> uc-muon -> lc-soap -> lc-adamw
# -> lc-muon).
#
# Skips any script whose output already has a `_final.pt` (already
# trained to completion). A script with a `_resume.pt` but no
# `_final.pt` (interrupted mid-run) is re-launched and resumes on its
# own via that script's existing checkpoint/resume logic -- no special
# handling needed here for that case.
#
# To stop: Ctrl+C at any time. Whatever script is currently running gets
# interrupted the same as any manual run (leaves its own `_resume.pt`
# behind, nothing corrupted -- see v3_CHANGELOG.md).
# To resume later: run this exact same script again. It re-checks what's
# done, skips those, and picks back up starting with the first
# not-yet-completed script in the list.
#
# Scripts live in digit_models/, router_models/, uppercase_models/, and
# lowercase_models/ (2026-08-02 reorganization, see v3_CHANGELOG.md) --
# $scripts entries below are subfolder-qualified paths, not bare filenames.
#
# python.exe path below is specific to this machine (per this project's
# own convention -- see supplementary_data.py's own path comments).

$python = "C:\Users\Will\AppData\Local\Python\pythoncore-3.14-64\python.exe"
$root   = "E:\mnist_v3"

$scripts = @(
    "digit_models\v3_mnist_digit_soap_16.py",  "digit_models\v3_mnist_digit_adamw_16.py",  "digit_models\v3_mnist_digit_muon_16.py",  "router_models\v3_mnist_router_ranger_16.py",
    "digit_models\v3_mnist_digit_soap_28.py",  "digit_models\v3_mnist_digit_adamw_28.py",  "digit_models\v3_mnist_digit_muon_28.py",  "router_models\v3_mnist_router_ranger_28.py",
    "digit_models\v3_mnist_digit_soap_32.py",  "digit_models\v3_mnist_digit_adamw_32.py",  "digit_models\v3_mnist_digit_muon_32.py",  "router_models\v3_mnist_router_ranger_32.py",
    "digit_models\v3_mnist_digit_soap_64.py",  "digit_models\v3_mnist_digit_adamw_64.py",  "digit_models\v3_mnist_digit_muon_64.py",  "router_models\v3_mnist_router_ranger_64.py",
    "digit_models\v3_mnist_digit_soap_128.py", "digit_models\v3_mnist_digit_adamw_128.py", "digit_models\v3_mnist_digit_muon_128.py", "router_models\v3_mnist_router_ranger_128.py",

    "uppercase_models\v3_mnist_letter_uc_soap_28.py",  "uppercase_models\v3_mnist_letter_uc_adamw_28.py",  "uppercase_models\v3_mnist_letter_uc_muon_28.py",  "lowercase_models\v3_mnist_letter_lc_soap_28.py",  "lowercase_models\v3_mnist_letter_lc_adamw_28.py",  "lowercase_models\v3_mnist_letter_lc_muon_28.py",
    "uppercase_models\v3_mnist_letter_uc_soap_32.py",  "uppercase_models\v3_mnist_letter_uc_adamw_32.py",  "uppercase_models\v3_mnist_letter_uc_muon_32.py",  "lowercase_models\v3_mnist_letter_lc_soap_32.py",  "lowercase_models\v3_mnist_letter_lc_adamw_32.py",  "lowercase_models\v3_mnist_letter_lc_muon_32.py",
    "uppercase_models\v3_mnist_letter_uc_soap_64.py",  "uppercase_models\v3_mnist_letter_uc_adamw_64.py",  "uppercase_models\v3_mnist_letter_uc_muon_64.py",  "lowercase_models\v3_mnist_letter_lc_soap_64.py",  "lowercase_models\v3_mnist_letter_lc_adamw_64.py",  "lowercase_models\v3_mnist_letter_lc_muon_64.py",
    "uppercase_models\v3_mnist_letter_uc_soap_128.py", "uppercase_models\v3_mnist_letter_uc_adamw_128.py", "uppercase_models\v3_mnist_letter_uc_muon_128.py", "lowercase_models\v3_mnist_letter_lc_soap_128.py", "lowercase_models\v3_mnist_letter_lc_adamw_128.py", "lowercase_models\v3_mnist_letter_lc_muon_128.py"
)

foreach ($script in $scripts) {
    # $script is now subfolder-qualified (e.g. "digit_models\v3_mnist_digit_soap_16.py")
    # since the 2026-08-02 reorganization into digit_models/router_models/
    # uppercase_models/lowercase_models -- each script's own OUTPUT_ROOT
    # (Path(__file__).resolve().parent / f"...") creates its output dir
    # NEXT TO the script, inside that same subfolder, not directly under
    # $root -- so the checkpoint path must be derived from the script's
    # own directory, not $root directly (a bare Join-Path $root
    # "$baseName\..." would double the subfolder into the path).
    $scriptPath = Join-Path $root $script
    $scriptDir  = Split-Path $scriptPath -Parent
    $leafName   = [System.IO.Path]::GetFileNameWithoutExtension($script)
    $finalCkpt  = Join-Path $scriptDir "$leafName\${leafName}_final.pt"

    if (Test-Path $finalCkpt) {
        Write-Host "[Skip] $script -- already complete" -ForegroundColor Yellow
        continue
    }

    Write-Host "[Run] $script" -ForegroundColor Cyan
    & $python $scriptPath

    if ($LASTEXITCODE -ne 0) {
        Write-Host "[Stopped] $script exited with code $LASTEXITCODE (Ctrl+C or a real error). Re-run this script later to resume from here." -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

Write-Host "[Done] All 44 scripts complete." -ForegroundColor Green
