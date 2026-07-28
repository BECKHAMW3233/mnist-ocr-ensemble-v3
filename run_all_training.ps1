# run_all_training.ps1
# =====================
# Runs all 20 v3 training scripts in order: 16x16 -> 28x28 -> 32x32 ->
# 64x64 -> 128x128, each resolution as soap -> adamw -> muon -> router.
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
# python.exe path below is specific to this machine (per this project's
# own convention -- see supplementary_data.py's own path comments).

$python = "C:\Users\Will\AppData\Local\Python\pythoncore-3.14-64\python.exe"
$root   = "E:\mnist_v3"

$scripts = @(
    "v3_mnist_digit_soap_16.py",  "v3_mnist_digit_adamw_16.py",  "v3_mnist_digit_muon_16.py",  "v3_mnist_router_ranger_16.py",
    "v3_mnist_digit_soap_28.py",  "v3_mnist_digit_adamw_28.py",  "v3_mnist_digit_muon_28.py",  "v3_mnist_router_ranger_28.py",
    "v3_mnist_digit_soap_32.py",  "v3_mnist_digit_adamw_32.py",  "v3_mnist_digit_muon_32.py",  "v3_mnist_router_ranger_32.py",
    "v3_mnist_digit_soap_64.py",  "v3_mnist_digit_adamw_64.py",  "v3_mnist_digit_muon_64.py",  "v3_mnist_router_ranger_64.py",
    "v3_mnist_digit_soap_128.py", "v3_mnist_digit_adamw_128.py", "v3_mnist_digit_muon_128.py", "v3_mnist_router_ranger_128.py"
)

foreach ($script in $scripts) {
    $baseName  = $script -replace '\.py$', ''
    $finalCkpt = Join-Path $root "$baseName\${baseName}_final.pt"

    if (Test-Path $finalCkpt) {
        Write-Host "[Skip] $script -- already complete" -ForegroundColor Yellow
        continue
    }

    Write-Host "[Run] $script" -ForegroundColor Cyan
    & $python (Join-Path $root $script)

    if ($LASTEXITCODE -ne 0) {
        Write-Host "[Stopped] $script exited with code $LASTEXITCODE (Ctrl+C or a real error). Re-run this script later to resume from here." -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

Write-Host "[Done] All 20 scripts complete." -ForegroundColor Green
