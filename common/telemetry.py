"""
common/telemetry.py
====================
Hardware telemetry — nvidia-smi-based GPU stats, psutil-based CPU/RAM/disk
stats, logged to the per-run CSV.

v3 telemetry expansion (2026-07-28, per direct user follow-up): the
original get_hw_stats() (kept below, unchanged, for any caller that still
wants a single cheap snapshot) sampled ONE snapshot per epoch at the
midpoint batch. A real run's own transcript showed this landing on an
idle spot more than once (CUDA 0% at one epoch, 5% at another) — a
single point sample can catch a validation/checkpoint lull instead of
representing the epoch's real GPU utilization. HardwareMonitor below
replaces that pattern for callers that want it: sample() multiple times
across an epoch, epoch_summary() reduces those samples to min/avg/max
per metric (so both "how low did it dip" and "what's the real average"
are visible, not just one number), plus adds power draw, GPU clock
speeds, throttle-reason flags, GPU memory-bandwidth utilization, fan
speed, and disk I/O throughput — none of which the original
get_hw_stats() captured.

Caller-side timing (data-wait vs. compute time per epoch) is NOT part of
this module — it's measured directly in each training script's
train_one_epoch() around the dataloader iteration, since only the caller
knows where the "waiting on the next batch" boundary actually is for its
own training-step call shape (SOAP's plain backward vs. every other
optimizer's amp_train_step()).

NOTE ON VALIDATION: the new nvidia-smi fields below (power.draw,
clocks.sm/mem, fan.speed, clocks_throttle_reasons.*) are documented,
standard nvidia-smi --query-gpu fields, but — unlike utilization.gpu/
temperature.gpu, which this project has confirmed working against real
hardware (RTX 4080) in prior transcripts — these specific fields have
NOT yet been confirmed against a real run in this project. Parsing is
defensive (each field falls back to -1 / "unknown" independently if
missing, "N/A", or the whole nvidia-smi call fails) so a field this
particular card doesn't expose degrades to -1 rather than crashing the
whole sample, but the actual values these produce on your hardware
should be checked against a real run before being trusted in a report.
"""
import os
import subprocess
import time

import torch

try:
    import psutil
    psutil.cpu_percent()  # verify it actually works, not just imports
    HAS_PSUTIL = True
except Exception:
    HAS_PSUTIL = False


def _cuda_index() -> int:
    """
    Returns the physical GPU index nvidia-smi queries should target
    (2026-07-28, per direct user follow-up — multi-GPU support). Every
    nvidia-smi --query-gpu call in this file used to omit -i entirely,
    which queries EVERY GPU on the system and returns one CSV row per
    card — on a single-GPU machine this happened to parse correctly by
    accident (exactly one row), but on any machine with 2+ GPUs it would
    have silently mis-parsed multi-row output as a single row. Uses
    torch.cuda.current_device() — set once via setup_device()'s new
    gpu_id parameter (torch.cuda.set_device()) at the very start of a
    run — so every telemetry call in this file automatically targets
    whichever GPU this process actually selected, without needing the
    GPU index threaded through every function signature.
    """
    return torch.cuda.current_device() if torch.cuda.is_available() else 0


def get_hw_stats() -> dict:
    """
    Original single-snapshot telemetry — GPU util/temp via nvidia-smi,
    VRAM peaks via torch.cuda, CPU/RAM via psutil. Kept as-is (unchanged
    field names/behavior) for any caller that only needs one cheap
    reading rather than HardwareMonitor's multi-sample min/avg/max.
    """
    stats = {
        "vram_peak_alloc_gb":    round(torch.cuda.max_memory_allocated() / 1024**3, 3) if torch.cuda.is_available() else -1,
        "vram_peak_reserved_gb": round(torch.cuda.max_memory_reserved()  / 1024**3, 3) if torch.cuda.is_available() else -1,
        "cuda_util_pct": -1, "gpu_temp_c": -1,
        "cpu_pct": -1, "ram_used_gb": -1, "ram_total_gb": -1,
    }
    try:
        result = subprocess.run(
            ["nvidia-smi", f"-i={_cuda_index()}",
             "--query-gpu=utilization.gpu,temperature.gpu",
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


# nvidia-smi --query-gpu field list for HardwareMonitor.sample(), one
# subprocess call gathering everything at once (not one call per field —
# calling nvidia-smi several times per epoch is already borderline
# expensive; multiplying that by the number of new fields would not be).
_NVSMI_FIELDS = (
    "utilization.gpu,utilization.memory,temperature.gpu,power.draw,"
    "clocks.sm,clocks.mem,fan.speed,"
    "clocks_throttle_reasons.hw_thermal_slowdown,"
    "clocks_throttle_reasons.sw_thermal_slowdown,"
    "clocks_throttle_reasons.hw_power_brake_slowdown,"
    "memory.used,memory.total"
)
_NVSMI_FIELD_COUNT = len(_NVSMI_FIELDS.split(","))


def _parse_nvsmi_float(raw: str):
    raw = raw.strip()
    if not raw or raw.upper() == "N/A":
        return -1
    try:
        return float(raw)
    except ValueError:
        return -1


def _parse_nvsmi_throttle(raw: str) -> int:
    """'Active' -> 1, 'Not Active' -> 0, anything else/missing -> -1 (unknown)."""
    raw = raw.strip()
    if raw == "Active":
        return 1
    if raw == "Not Active":
        return 0
    return -1


class HardwareMonitor:
    """
    Per-epoch (or per-run) hardware telemetry sampler. Instantiate once
    per epoch in train_one_epoch(), call .sample() at several points
    across the epoch (not just the midpoint), then reduce the collected
    samples with the static .epoch_summary() before returning from the
    epoch — see any v3 training script's train_one_epoch() for the real
    usage pattern.

    Disk I/O is reported as a rate (MB/s), not a cumulative total —
    psutil.disk_io_counters() itself only returns bytes-since-boot, so
    each .sample() call computes the delta against this instance's own
    previous sample (or against instantiation time, for the first
    sample) divided by the elapsed wall-clock time. This means disk
    throughput numbers are only meaningful relative to calls on the SAME
    HardwareMonitor instance — do not compare a rate from one instance's
    first sample against another instance's first sample and expect it
    to mean the same thing across different elapsed windows.
    """

    def __init__(self):
        self._last_disk = psutil.disk_io_counters() if HAS_PSUTIL else None
        self._last_time = time.time()

    def sample(self) -> dict:
        now = time.time()
        elapsed = max(now - self._last_time, 1e-6)

        stats = {
            "vram_alloc_gb":     round(torch.cuda.max_memory_allocated() / 1024**3, 3) if torch.cuda.is_available() else -1,
            "vram_reserved_gb":  round(torch.cuda.max_memory_reserved()  / 1024**3, 3) if torch.cuda.is_available() else -1,
            "cuda_util_pct":     -1,
            "cuda_mem_util_pct": -1,
            "gpu_temp_c":        -1,
            "gpu_power_w":       -1,
            "gpu_clock_sm_mhz":  -1,
            "gpu_clock_mem_mhz": -1,
            "gpu_fan_pct":       -1,
            "gpu_throttled":     -1,  # -1 unknown, 0 not throttled, 1 throttled
            "cpu_pct":           -1,
            "ram_used_gb":       -1,
            "ram_total_gb":      -1,
            "disk_read_mb_s":    -1,
            "disk_write_mb_s":   -1,
            "nvsmi_vram_used_gb":  -1,
            "nvsmi_vram_total_gb": -1,
        }

        try:
            result = subprocess.run(
                ["nvidia-smi", f"-i={_cuda_index()}",
                 f"--query-gpu={_NVSMI_FIELDS}",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            parts = result.stdout.strip().split(",")
            if len(parts) == _NVSMI_FIELD_COUNT:
                stats["cuda_util_pct"]     = int(_parse_nvsmi_float(parts[0]))
                stats["cuda_mem_util_pct"] = int(_parse_nvsmi_float(parts[1]))
                stats["gpu_temp_c"]        = int(_parse_nvsmi_float(parts[2]))
                stats["gpu_power_w"]       = round(_parse_nvsmi_float(parts[3]), 1)
                stats["gpu_clock_sm_mhz"]  = int(_parse_nvsmi_float(parts[4]))
                stats["gpu_clock_mem_mhz"] = int(_parse_nvsmi_float(parts[5]))
                stats["gpu_fan_pct"]       = int(_parse_nvsmi_float(parts[6]))
                _throttle_flags = [_parse_nvsmi_throttle(p) for p in parts[7:10]]
                if any(f == 1 for f in _throttle_flags):
                    stats["gpu_throttled"] = 1
                elif all(f == 0 for f in _throttle_flags):
                    stats["gpu_throttled"] = 0
                # else: mixed/unknown flags -> stays -1
                _used_mb  = _parse_nvsmi_float(parts[10])
                _total_mb = _parse_nvsmi_float(parts[11])
                stats["nvsmi_vram_used_gb"]  = round(_used_mb / 1024, 3)  if _used_mb  != -1 else -1
                stats["nvsmi_vram_total_gb"] = round(_total_mb / 1024, 3) if _total_mb != -1 else -1
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
            try:
                cur_disk = psutil.disk_io_counters()
                if cur_disk is not None and self._last_disk is not None:
                    stats["disk_read_mb_s"]  = round((cur_disk.read_bytes  - self._last_disk.read_bytes)  / 1024**2 / elapsed, 2)
                    stats["disk_write_mb_s"] = round((cur_disk.write_bytes - self._last_disk.write_bytes) / 1024**2 / elapsed, 2)
                self._last_disk = cur_disk
            except Exception:
                pass

        self._last_time = now
        return stats

    @staticmethod
    def epoch_summary(samples: list) -> dict:
        """
        Reduces a list of .sample() dicts (at least one) into a single
        per-epoch summary: min/avg/max for volatile metrics (things that
        genuinely swing during an epoch — utilization, temp, power, CPU,
        RAM used, disk throughput), avg-only for clock speeds and fan speed
        (less volatile, avg is representative), max for the throttle
        flag (1 if throttled at ANY sampled point during the epoch, not
        just the last one), and last-value for things that don't change
        within a run (ram_total_gb) or that are already cumulative peaks
        (vram_alloc_gb/vram_reserved_gb, via torch.cuda.max_memory_*).
        Ignores -1 (unknown) readings when computing min/avg/max for a
        field — a field entirely unavailable this run reports -1 for all
        three rather than being dragged down by treating -1 as a real 0.
        """
        def _valid(field):
            return [s[field] for s in samples if s.get(field, -1) != -1]

        def _min_avg_max(field):
            vals = _valid(field)
            if not vals:
                return -1, -1, -1
            return min(vals), round(sum(vals) / len(vals), 2), max(vals)

        summary = {}
        for field in ("cuda_util_pct", "cuda_mem_util_pct", "gpu_temp_c",
                      "gpu_power_w", "cpu_pct", "ram_used_gb",
                      "disk_read_mb_s", "disk_write_mb_s", "nvsmi_vram_used_gb"):
            lo, avg, hi = _min_avg_max(field)
            summary[f"{field}_min"] = lo
            summary[f"{field}_avg"] = avg
            summary[f"{field}_max"] = hi

        for field in ("gpu_clock_sm_mhz", "gpu_clock_mem_mhz", "gpu_fan_pct"):
            vals = _valid(field)
            summary[f"{field}_avg"] = round(sum(vals) / len(vals), 1) if vals else -1

        throttle_vals = [s.get("gpu_throttled", -1) for s in samples]
        if any(v == 1 for v in throttle_vals):
            summary["gpu_throttled_any"] = 1
        elif any(v == 0 for v in throttle_vals):
            summary["gpu_throttled_any"] = 0
        else:
            summary["gpu_throttled_any"] = -1

        summary["ram_total_gb"]          = samples[-1].get("ram_total_gb", -1)
        summary["vram_peak_alloc_gb"]    = samples[-1].get("vram_alloc_gb", -1)
        summary["vram_peak_reserved_gb"] = samples[-1].get("vram_reserved_gb", -1)
        summary["nvsmi_vram_total_gb"]   = samples[-1].get("nvsmi_vram_total_gb", -1)
        return summary


def _format_vram_trend(vram_history: list, window: int = 5) -> str:
    """
    Formats the last `window` epochs' peak VRAM readings from vram_history
    (history['nvsmi_vram_used_gb_max'], includes the current epoch as its
    last entry) into a human-readable trend line — the raw sequence plus
    the average epoch-over-epoch delta across the window's earlier epochs
    versus the jump into the current (last) epoch. No sudden-vs-gradual
    verdict — deliberately just the numbers, so whoever reads the log
    judges the shape of the curve themselves (2026-08-08, per direct user
    follow-up: an algorithmic label would need an invented threshold for
    what counts as "sudden," which has no principled basis). Returns ""
    if there isn't enough history yet to show anything meaningful.
    """
    recent = vram_history[-window:]
    if len(recent) < 2:
        return ""
    sequence = " -> ".join(f"{v:.2f}" for v in recent)
    start_epoch = len(vram_history) - len(recent) + 1
    end_epoch = len(vram_history)
    line = f"  [VRAM] Recent trend (epochs {start_epoch}-{end_epoch}): {sequence} GB"
    if len(recent) >= 3:
        prior_deltas = [recent[i] - recent[i - 1] for i in range(1, len(recent) - 1)]
        avg_prior_delta = sum(prior_deltas) / len(prior_deltas)
        this_epoch_delta = recent[-1] - recent[-2]
        line += (f" (avg {avg_prior_delta:+.2f}GB/epoch over epochs "
                 f"{start_epoch}-{end_epoch - 1}, this epoch {this_epoch_delta:+.2f}GB)")
    return line


def check_vram_safety(nvsmi_used_peak_gb: float, nvsmi_total_gb: float, epoch: int,
                       warn_reserve_gb: float = 1.0, stop_reserve_gb: float = 0.25,
                       vram_history: list = None) -> str:
    """
    Returns "stop", "warn", or "ok" — the epoch's PEAK nvidia-smi VRAM
    usage (nvsmi_used_peak_gb, from HardwareMonitor.epoch_summary()'s
    nvsmi_vram_used_gb_max, itself reduced from the same interval-sampled
    HardwareMonitor.sample() calls every other telemetry field uses — see
    that function's field-list comment) compared against this GPU's real
    physical capacity (nvsmi_total_gb, from epoch_summary()'s
    nvsmi_vram_total_gb). "ok" (no-op) if either value is unavailable
    (< 0). Prints a warning first for "warn"/"stop".

    vram_history (2026-08-08, per direct user follow-up): optional list of
    every prior epoch's peak VRAM (history['nvsmi_vram_used_gb_max'],
    including this epoch as its last entry, since the caller's history
    dict is updated before this function is called each epoch) — when
    given, a "warn" or "stop" print is followed by a trend line (see
    _format_vram_trend()) showing the last few epochs' raw readings and
    the recent per-epoch delta versus this epoch's own jump, so it's
    visible from the log whether this was a gradual creep or a sudden
    single-epoch spike. No-op (omitted) if not given.

    Two-tier by design (2026-08-08, per direct user follow-up): the
    original single reserve_gb=1.0 line was being enforced as a hard
    stop, but its actual intent was always advisory — training should
    try to avoid eating into that buffer, and it's fine if a given epoch
    does, since the point is to stop the GPU from being consumed so
    completely that it bottlenecks the rest of the system, not to treat
    1GB of headroom as a crash boundary. warn_reserve_gb keeps that
    original 1.0 GB line as a signal the caller acts on by reactively
    reducing batch size (see common/batch_sizing.py's
    reduce_batch_size()) and continuing, not stopping. stop_reserve_gb
    is the actual hard-stop trigger, moved much closer to real
    exhaustion. Changed from a bool to a 3-state string return
    (2026-08-08) because "warn" now needs to carry caller-actionable
    meaning ("reduce batch size and continue") distinct from both "ok"
    (do nothing) and "stop" (halt) — a bool could no longer represent
    the three distinct outcomes.

    2026-08-07 fix, two rounds: first, this compared torch.cuda's own
    max_memory_reserved() figure against nvidia-smi's total — mixing two
    measurement systems this project's own VRAM investigation showed can
    disagree (torch.cuda's reserved figure can read higher than physical
    capacity on Windows/WDDM). A real run then proved this concretely:
    nvidia-smi logged 8.953 GB used against a 15.992 GB card while
    torch.cuda's reserved figure claimed 15.012 GB, triggering a false
    stop with ~7 GB of real headroom untouched. Fixed by switching both
    sides of the comparison to nvidia-smi — but that first fix still took
    only one nvidia-smi reading, right after the epoch ended, missing any
    genuine mid-epoch spike that dropped back down before that single
    check ran. This project's own HardwareMonitor already exists
    specifically to avoid exactly that failure mode for every other
    metric (see this module's docstring: a single end-of-epoch/midpoint
    snapshot for cuda_util_pct was already shown to land on an idle spot
    and miss the epoch's real behavior) — VRAM just wasn't wired into
    that interval-sampled path yet. It is now: nvidia-smi's memory.used/
    memory.total ride along on the same _NVSMI_FIELDS call every
    HardwareMonitor.sample() already makes, so this check sees the real
    peak across the epoch, not a single snapshot.
    """
    if nvsmi_total_gb < 0 or nvsmi_used_peak_gb < 0:
        return "ok"
    if nvsmi_used_peak_gb > (nvsmi_total_gb - stop_reserve_gb):
        print(f"  [VRAM] {nvsmi_used_peak_gb:.2f} GB peak this epoch — within "
              f"{stop_reserve_gb:.2f} GB of this GPU's {nvsmi_total_gb:.2f} GB "
              f"physical capacity — stopping training cleanly after epoch {epoch}")
        if vram_history:
            _trend = _format_vram_trend(vram_history)
            if _trend:
                print(_trend)
        return "stop"
    if nvsmi_used_peak_gb > (nvsmi_total_gb - warn_reserve_gb):
        print(f"  [VRAM] {nvsmi_used_peak_gb:.2f} GB peak this epoch — within "
              f"{warn_reserve_gb:.1f} GB of this GPU's {nvsmi_total_gb:.2f} GB "
              f"physical capacity — reducing batch size and continuing")
        if vram_history:
            _trend = _format_vram_trend(vram_history)
            if _trend:
                print(_trend)
        return "warn"
    return "ok"


def check_cpu_ram_safety(device: torch.device, swap_baseline_gb: float,
                          ram_reserve_gb: float, epoch: int) -> bool:
    """
    Returns True if CPU-only training should stop — either swap usage
    grew past this run's own baseline (treated as OOM) or free RAM
    dropped below the configured reserve (2026-08-07, per direct user
    follow-up — centralized here from the identical block previously
    duplicated across all 44 training scripts, same as
    check_vram_safety() above). No-op (False) on a GPU run — this check
    is specifically about CPU-only training exhausting system RAM/swap;
    a GPU run's own swap fluctuation is an unrelated signal and must not
    stop a healthy CUDA run (2026-08-07 fix — this device-type gate was
    lost when the original per-script `if device.type == "cpu"` block
    was centralized into this function, and fired incorrectly on a real
    GPU training run as a result — see v3_CHANGELOG.md). Prints a
    warning first. No-op (False) if psutil isn't available.
    """
    if device.type != "cpu" or not HAS_PSUTIL:
        return False
    _swap_used_gb = round(psutil.swap_memory().used / 1024**3, 3)
    _free_ram_gb  = psutil.virtual_memory().available / 1024**3
    if _swap_used_gb > swap_baseline_gb:
        print(f"  [RAM] Page file / swap usage GREW beyond this run's "
              f"{swap_baseline_gb:.3f} GB baseline ({_swap_used_gb:.3f} GB now) — "
              f"treating as OOM, stopping training cleanly after epoch {epoch}")
        return True
    if _free_ram_gb < ram_reserve_gb:
        print(f"  [RAM] Free RAM ({_free_ram_gb:.2f} GB) below the "
              f"{ram_reserve_gb} GB reserve — stopping training "
              f"cleanly after epoch {epoch}")
        return True
    return False


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
        _idx = _cuda_index()
        _smi = subprocess.check_output(
            ['nvidia-smi', f'-i={_idx}', '--query-gpu=memory.used,memory.total',
             '--format=csv,noheader,nounits'],
            timeout=3
        ).decode().strip()
        _used_mb, _total_mb = (float(x.strip()) for x in _smi.split(','))
        print(f"[Startup VRAM] GPU {_idx}: {_used_mb:.0f} MB / {_total_mb:.0f} MB in use "
              f"before this script has allocated anything")
        if _used_mb > 500:
            try:
                # observed (not independently verified, unlike the WDDM
                # claim above): --query-compute-apps appears to have no
                # per-GPU -i filter in older nvidia-smi builds on some
                # systems — queried across all
                # GPUs and left unfiltered by index; on a multi-GPU
                # machine this list may include processes on OTHER cards
                # too, not just the one this run selected. Still useful
                # as a "something is holding VRAM somewhere" signal.
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


def setup_device(cpu_reserve_pct: int = 25, use_amp: bool = True, gpu_id: int = None) -> torch.device:
    """
    Resolves and reports the training device, and reports CPU core
    reservation / DataLoader worker count (2026-07-28, per direct user
    follow-up) — always printed now, not just on CPU-only runs, since
    NUM_WORKERS (every training script's DataLoader worker count) uses
    this exact same usable-core number regardless of whether model
    compute happens on GPU or CPU; before this fix, a GPU run printed
    nothing at all about core count or worker allocation. reserve_cpu_
    threads() (common/seeding.py) must already have been called
    immediately after `import torch` — this function only reports the
    thread count it set, it doesn't set it itself (torch.set_num_threads()
    must be called before set_all_seeds()'s own torch calls to reliably
    take effect, so the actual reservation happens earlier than this
    function runs).

    gpu_id (2026-07-28, per direct user follow-up — multi-GPU support):
    optional physical GPU index to train on, from each script's --gpu
    CLI flag. Calling torch.cuda.set_device(gpu_id) HERE, before anything
    else in this function runs, is what makes every other GPU-index-aware
    call in this codebase (_cuda_index() in this module, and the
    get_device_properties()/nvidia-smi calls that use it — including
    common/batch_sizing.py's probe) automatically target the right card
    without gpu_id being threaded through every single function
    signature: torch.cuda.current_device() reflects this call for the
    rest of the process's lifetime. None (the default) leaves whatever
    device CUDA already considers current untouched — correct for a
    single-GPU machine, or for a multi-GPU machine run under `torchrun`/
    DDP (see common/distributed.py), where the launcher itself already
    sets the correct per-process device before this function is called.
    """
    if gpu_id is not None and torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)

    total_cores = os.cpu_count() or 1
    reserved    = max(1, round(total_cores * cpu_reserve_pct / 100))
    usable      = max(1, total_cores - reserved)
    print(f"[CPU] {total_cores} logical cores detected — reserving {reserved} "
          f"({cpu_reserve_pct}%) for the OS, {usable} usable "
          f"(this machine's NUM_WORKERS = {usable} DataLoader workers; "
          f"also the thread count torch.set_num_threads() uses on CPU-only runs)")

    if torch.cuda.is_available():
        _idx   = _cuda_index()
        device = torch.device(f"cuda:{_idx}")
        props  = torch.cuda.get_device_properties(_idx)
        vram   = props.total_memory / 1024**3
        _gpu_count = torch.cuda.device_count()
        _selection_note = f" (--gpu {gpu_id} selected)" if gpu_id is not None else ""
        print(f"[Device] GPU {_idx}/{_gpu_count - 1}: {props.name}  |  {vram:.1f} GB VRAM  |  "
              f"CUDA {torch.version.cuda}  |  AMP: {use_amp}{_selection_note}")
        if _gpu_count > 1 and gpu_id is None:
            print(f"[Device] {_gpu_count} GPUs detected on this machine — training will use "
                  f"GPU {_idx} (CUDA's current default) since no --gpu flag was passed. "
                  f"Pass --gpu N to pick a specific card explicitly, e.g. to run two "
                  f"different scripts on two different GPUs at once.")
        print_vram_baseline()
    else:
        device = torch.device("cpu")
        print(f"[Device] CPU — training compute uses {usable} of {total_cores} threads")
    return device
