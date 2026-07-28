"""
common/telemetry.py
====================
Hardware telemetry — nvidia-smi-based GPU stats plus psutil-based CPU/RAM
stats, sampled once per epoch (at the midpoint batch, so the reading
reflects the GPU while it's actively training rather than idling or
winding down post-loop) and logged to the per-run CSV. Extracted
verbatim from the identical get_hw_stats() / print_vram_baseline() /
setup_device() trio duplicated across every v2 training script.
"""
import os
import subprocess

import torch

try:
    import psutil
    psutil.cpu_percent()  # verify it actually works, not just imports
    HAS_PSUTIL = True
except Exception:
    HAS_PSUTIL = False


def get_hw_stats() -> dict:
    """Per-epoch hardware telemetry — GPU via torch.cuda + nvidia-smi, CPU/RAM via psutil."""
    stats = {
        "vram_peak_alloc_gb":    round(torch.cuda.max_memory_allocated() / 1024**3, 3) if torch.cuda.is_available() else -1,
        "vram_peak_reserved_gb": round(torch.cuda.max_memory_reserved()  / 1024**3, 3) if torch.cuda.is_available() else -1,
        "cuda_util_pct": -1, "gpu_temp_c": -1,
        "cpu_pct": -1, "ram_used_gb": -1, "ram_total_gb": -1,
    }
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        parts = result.stdout.strip().split(",")
        stats["cuda_util_pct"] = int(parts[0].strip())
        stats["gpu_temp_c"]    = int(parts[1].strip())
    except Exception:
        pass
    if HAS_PSUTIL:
        try:
            stats["cpu_pct"]      = psutil.cpu_percent(interval=None)
            vm = psutil.virtual_memory()
            stats["ram_used_gb"]  = round(vm.used  / 1024**3, 2)
            stats["ram_total_gb"] = round(vm.total / 1024**3, 2)
        except Exception:
            pass
    return stats


def print_vram_baseline() -> None:
    """
    Best-effort report of VRAM already in use before this script allocates
    anything — torch.cuda.empty_cache() (called between batch-size probe
    candidates) can only free memory this process's own PyTorch has
    cached but isn't using; it cannot see memory held by other processes.
    Per-process attribution (--query-compute-apps) is NOT available on
    Windows under WDDM (the default driver mode for GeForce cards,
    confirmed via NVIDIA's own nvidia-smi docs) — those rows come back as
    literal "N/A" and are filtered out here rather than printed verbatim.
    Silently prints nothing extra if nvidia-smi isn't available or times
    out.
    """
    try:
        _smi = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.used,memory.total',
             '--format=csv,noheader,nounits'],
            timeout=3
        ).decode().strip()
        _used_mb, _total_mb = (float(x.strip()) for x in _smi.split(','))
        print(f"[Startup VRAM] {_used_mb:.0f} MB / {_total_mb:.0f} MB in use "
              f"before this script has allocated anything")
        if _used_mb > 500:
            try:
                _procs = subprocess.check_output(
                    ['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
                     '--format=csv,noheader,nounits'],
                    timeout=3
                ).decode().strip()
                _real_procs = [
                    _line.strip() for _line in _procs.splitlines()
                    if _line.strip() and "N/A" not in _line
                ]
                if _real_procs:
                    print("[Startup VRAM] Processes already using GPU memory:")
                    for _line in _real_procs:
                        print(f"    {_line}")
                    print("[Startup VRAM] If this run's batch size looks lower than "
                          "expected, check whether these processes are why.")
                elif _procs.strip():
                    print("[Startup VRAM] Something is using GPU memory, but Windows' "
                          "WDDM driver mode doesn't expose which process to nvidia-smi "
                          "(check Task Manager's Performance > GPU tab instead).")
            except subprocess.SubprocessError:
                pass
    except subprocess.SubprocessError:
        pass  # best-effort only


def setup_device(cpu_reserve_pct: int = 25, use_amp: bool = True) -> torch.device:
    """
    Resolves and reports the training device. reserve_cpu_threads()
    (common/seeding.py) must already have been called immediately after
    `import torch` — this function only reports the thread count it set,
    it doesn't set it itself (torch.set_num_threads() must be called
    before set_all_seeds()'s own torch calls to reliably take effect, so
    the actual reservation happens earlier than this function runs).
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        props  = torch.cuda.get_device_properties(0)
        vram   = props.total_memory / 1024**3
        print(f"[Device] {props.name}  |  {vram:.1f} GB VRAM  |  "
              f"CUDA {torch.version.cuda}  |  AMP: {use_amp}")
        print_vram_baseline()
    else:
        device = torch.device("cpu")
        total_cores = os.cpu_count() or 1
        reserved    = max(1, round(total_cores * cpu_reserve_pct / 100))
        usable      = max(1, total_cores - reserved)
        print(f"[Device] CPU — {total_cores} logical cores detected, "
              f"reserving {reserved} for Windows ({cpu_reserve_pct}%), "
              f"using {usable}")
    return device
