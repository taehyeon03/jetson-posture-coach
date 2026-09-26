"""Benchmark instrumentation for the posture pipeline.

Standard library only (Python 3.6 compatible) so it imports on the Nano and
in unit tests without OpenCV/TensorRT. It has three parts:

* ``FrameRecorder`` -- receives one record per processed frame from
  ``web_mvp.server.worker`` (per-stage timings, detection flags, keypoints).
* ``SystemSampler`` -- background RSS / RAM / swap / temperature / power
  sampling (``tegrastats`` when available, ``/proc`` and sysfs otherwise).
* ``summarize`` -- turns both into the numbers reported by
  ``scripts/benchmark.py`` and checks them against
  ``docs/system_validation_plan.md``.
"""

from __future__ import print_function

import csv
import glob
import math
import os
import re
import shutil
import subprocess
import threading
import time

STAGES = (
    "capture_wait_ms",
    "age_ms",
    "face_ms",
    "infer_ms",
    "pre_ms",
    "h2d_ms",
    "exec_ms",
    "d2h_ms",
    "decode_ms",
    "track_ms",
    "state_ms",
    "encode_ms",
    "total_ms",
    "e2e_ms",
)

FRAME_COLUMNS = (
    "frame_id",
    "t_s",
) + STAGES + (
    "face_found",
    "pose_valid",
    "lock_ok",
    "state",
    "head_forward",
    "torso_forward",
    "jitter_raw_px",
    "jitter_smooth_px",
    "mean_score",
)

SAMPLE_COLUMNS = (
    "t_s",
    "phase",
    "rss_mb",
    "ram_used_mb",
    "swap_used_mb",
    "temp_max_c",
    "gpu_util_pct",
    "power_mw",
)

# Acceptance targets from docs/system_validation_plan.md section 1.
TARGET_P95_MS = 200.0
TARGET_FPS = 5.0
TARGET_RSS_MB = 3500.0


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------

def percentile(sorted_values, q):
    """Linear-interpolated percentile of an ascending list, ``q`` in [0, 100]."""
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    rank = (len(sorted_values) - 1) * q / 100.0
    low = int(math.floor(rank))
    high = int(math.ceil(rank))
    fraction = rank - low
    return float(sorted_values[low] + (sorted_values[high] - sorted_values[low]) * fraction)


def distribution(values):
    """count/mean/std/min/p50/p95/p99/max of the non-None ``values``."""
    data = sorted(float(v) for v in values if v is not None)
    if not data:
        return None
    mean = sum(data) / len(data)
    variance = sum((v - mean) ** 2 for v in data) / len(data)
    return {
        "count": len(data),
        "mean": round(mean, 3),
        "std": round(math.sqrt(variance), 3),
        "min": round(data[0], 3),
        "p50": round(percentile(data, 50), 3),
        "p95": round(percentile(data, 95), 3),
        "p99": round(percentile(data, 99), 3),
        "max": round(data[-1], 3),
    }


def linear_slope(xs, ys):
    """Least-squares slope of ``ys`` over ``xs`` (units of y per unit of x)."""
    n = len(xs)
    if n < 2:
        return None
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    denominator = sum((x - mean_x) ** 2 for x in xs)
    if denominator == 0:
        return None
    return sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denominator


def _rate(count, total):
    return None if not total else round(float(count) / total, 4)


# --------------------------------------------------------------------------
# Per-frame recording
# --------------------------------------------------------------------------

class FrameRecorder(object):
    """Collects the per-frame records the worker loop hands over.

    ``frame()`` is called from the worker thread only; the runner reads the
    lists after the worker has stopped.
    """

    def __init__(self, confidence=0.25):
        self.confidence = float(confidence)
        self.origin = time.monotonic()
        self.frames = []
        self.camera_timeouts = 0
        self.errors = []
        self.gpu_mem_used_mb = []
        self.gpu_mem_probe = None
        self.joint_names = None
        self._previous_raw = None
        self._previous_smooth = None

    def camera_timeout(self):
        self.camera_timeouts += 1

    def error(self, exc):
        self.errors.append("%s: %s" % (type(exc).__name__, exc))

    def probe_gpu_memory(self):
        """Sample device memory in the calling (CUDA-context) thread."""
        if self.gpu_mem_probe is None:
            return
        info = self.gpu_mem_probe()
        if info is not None:
            free, total = info
            self.gpu_mem_used_mb.append((total - free) / 1048576.0)

    def _jitter(self, previous, current):
        """Mean frame-to-frame displacement in px over joints confident in both."""
        if previous is None:
            return None
        moves = []
        for name, point in current.items():
            before = previous.get(name)
            if before is None:
                continue
            if point[2] >= self.confidence and before[2] >= self.confidence:
                moves.append(math.hypot(point[0] - before[0], point[1] - before[1]))
        return sum(moves) / len(moves) if moves else None

    def frame(self, record, raw_points=None, smooth_points=None):
        """Store ``record`` (dict keyed by ``FRAME_COLUMNS``) plus per-joint scores.

        ``raw_points``/``smooth_points`` map joint name -> (x_px, y_px, score).
        Jitter is only measured between consecutive person-locked frames, so a
        lock rejection does not show up as a giant jump.
        """
        record = dict(record)
        record["frame_id"] = len(self.frames)
        record["t_s"] = record["t"] - self.origin
        record.setdefault("jitter_raw_px", None)
        record.setdefault("jitter_smooth_px", None)
        record.setdefault("mean_score", None)
        if raw_points is not None:
            if self.joint_names is None:
                self.joint_names = list(raw_points.keys())
            record["scores"] = tuple(raw_points[n][2] for n in self.joint_names if n in raw_points)
            record["mean_score"] = sum(record["scores"]) / len(record["scores"]) if record["scores"] else None
            if record.get("lock_ok") and smooth_points is not None:
                record["jitter_raw_px"] = self._jitter(self._previous_raw, raw_points)
                record["jitter_smooth_px"] = self._jitter(self._previous_smooth, smooth_points)
                self._previous_raw = raw_points
                self._previous_smooth = smooth_points
            else:
                self._previous_raw = None
                self._previous_smooth = None
        self.frames.append(record)

    def write_frames_csv(self, path):
        with open(path, "w") as stream:
            writer = csv.writer(stream)
            writer.writerow(FRAME_COLUMNS)
            for record in self.frames:
                writer.writerow([_cell(record.get(column)) for column in FRAME_COLUMNS])


def _cell(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return "%.4f" % value
    return value


# --------------------------------------------------------------------------
# System sampling
# --------------------------------------------------------------------------

_TEGRA_RAM = re.compile(r"RAM (\d+)/(\d+)MB")
_TEGRA_SWAP = re.compile(r"SWAP (\d+)/(\d+)MB")
_TEGRA_GPU = re.compile(r"GR3D_FREQ (\d+)%")
_TEGRA_POWER = re.compile(r"\b(?:POM_5V_IN|VDD_IN)\s+(\d+)\s*(?:mW)?/(\d+)")
_TEGRA_TEMP = re.compile(r"\b(\w+)@(-?\d+(?:\.\d+)?)C")

# PMIC@100C is a fixed placeholder on Nano and unused sensors report -256C.
_IGNORED_SENSORS = ("pmic",)


def parse_tegrastats(line):
    """Parse one ``tegrastats`` line into a dict (missing fields are None).

    Works for both Nano (``POM_5V_IN 1897/1897``) and Orin
    (``VDD_IN 4162mW/4162mW``) formats. Power is the instantaneous first value.
    """
    parsed = {"gpu_util_pct": None, "power_mw": None, "temp_max_c": None, "temps": {}}
    match = _TEGRA_GPU.search(line)
    if match:
        parsed["gpu_util_pct"] = float(match.group(1))
    match = _TEGRA_POWER.search(line)
    if match:
        parsed["power_mw"] = float(match.group(1))
    for name, value in _TEGRA_TEMP.findall(line):
        value = float(value)
        if name.lower() in _IGNORED_SENSORS or value <= -100.0:
            continue
        parsed["temps"][name.lower()] = value
    if parsed["temps"]:
        parsed["temp_max_c"] = max(parsed["temps"].values())
    return parsed


def _read_kv_kb(path, keys):
    values = {}
    try:
        with open(path) as stream:
            for line in stream:
                key, _, rest = line.partition(":")
                if key in keys:
                    values[key] = float(rest.split()[0]) / 1024.0
    except (IOError, OSError, ValueError, IndexError):
        pass
    return values


def read_process_memory():
    """(rss_mb, peak_rss_mb) of this process from /proc, else (None, None)."""
    values = _read_kv_kb("/proc/self/status", ("VmRSS", "VmHWM"))
    return values.get("VmRSS"), values.get("VmHWM")


def read_system_memory():
    """(ram_used_mb, swap_used_mb) from /proc/meminfo, else (None, None)."""
    values = _read_kv_kb("/proc/meminfo", ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree"))
    if "MemTotal" not in values or "MemAvailable" not in values:
        return None, None
    ram = values["MemTotal"] - values["MemAvailable"]
    swap = values.get("SwapTotal", 0.0) - values.get("SwapFree", 0.0)
    return ram, swap


def read_thermal_zones():
    """Max temperature (C) over /sys/class/thermal zones, else None."""
    temps = []
    for path in glob.glob("/sys/class/thermal/thermal_zone*/temp"):
        try:
            with open(path) as stream:
                value = float(stream.read().strip()) / 1000.0
        except (IOError, OSError, ValueError):
            continue
        if value > -100.0:
            temps.append(value)
    return max(temps) if temps else None


def find_power_file():
    """Nano INA3221 total-input rail (mW) sysfs file, if this board has one."""
    pattern = "/sys/bus/i2c/drivers/ina3221x/*/iio:device*/in_power0_input"
    matches = sorted(glob.glob(pattern))
    return matches[0] if matches else None


def read_power_file(path):
    try:
        with open(path) as stream:
            return float(stream.read().strip())
    except (IOError, OSError, ValueError):
        return None


class SystemSampler(threading.Thread):
    """Samples process/system memory, temperature and power once per interval.

    Power and GPU utilisation come from ``tegrastats`` (or ``power_file``
    sysfs). Both are board-reported rail values, not wall-socket power.
    """

    def __init__(self, interval=1.0, power_file=None, use_tegrastats=True):
        threading.Thread.__init__(self)
        self.daemon = True
        self.interval = float(interval)
        self.power_file = power_file or find_power_file()
        self.samples = []
        self.phase = "idle"
        self.power_source = "power_file" if self.power_file else None
        self._stop_event = threading.Event()
        self._tegra_process = None
        self._tegra_latest = None
        self._tegra_time = 0.0
        self._lock = threading.Lock()
        if use_tegrastats:
            self._start_tegrastats()

    def _start_tegrastats(self):
        binary = shutil.which("tegrastats")
        if binary is None:
            return
        try:
            command = [binary, "--interval", str(max(100, int(self.interval * 1000)))]
            if shutil.which("stdbuf"):
                # tegrastats block-buffers stdout when piped; force line output.
                command = ["stdbuf", "-oL"] + command
            self._tegra_process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                universal_newlines=True,
            )
        except (OSError, ValueError):
            self._tegra_process = None
            return
        self.power_source = "tegrastats"
        reader = threading.Thread(target=self._read_tegrastats)
        reader.daemon = True
        reader.start()

    def _read_tegrastats(self):
        for line in self._tegra_process.stdout:
            parsed = parse_tegrastats(line)
            with self._lock:
                self._tegra_latest = parsed
                self._tegra_time = time.monotonic()

    def mark(self, phase):
        self.phase = phase

    def _sample(self):
        now = time.monotonic()
        rss, _ = read_process_memory()
        ram, swap = read_system_memory()
        with self._lock:
            tegra = self._tegra_latest if now - self._tegra_time <= 3.0 * self.interval else None
        temp = tegra["temp_max_c"] if tegra else None
        if temp is None:
            temp = read_thermal_zones()
        power = tegra["power_mw"] if tegra else None
        if power is None and self.power_file:
            power = read_power_file(self.power_file)
        self.samples.append({
            "t_s": now,
            "phase": self.phase,
            "rss_mb": rss,
            "ram_used_mb": ram,
            "swap_used_mb": swap,
            "temp_max_c": temp,
            "gpu_util_pct": tegra["gpu_util_pct"] if tegra else None,
            "power_mw": power,
        })

    def run(self):
        self._sample()
        while not self._stop_event.wait(self.interval):
            self._sample()

    def stop(self):
        self._stop_event.set()
        self.join(timeout=self.interval + 2.0)
        self._sample()
        if not any(s["power_mw"] is not None for s in self.samples):
            self.power_source = None  # e.g. tegrastats needs root here and produced nothing
        if self._tegra_process is not None:
            self._tegra_process.terminate()
            try:
                self._tegra_process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self._tegra_process.kill()

    def write_csv(self, path, origin=0.0):
        with open(path, "w") as stream:
            writer = csv.writer(stream)
            writer.writerow(SAMPLE_COLUMNS)
            for sample in self.samples:
                row = dict(sample)
                row["t_s"] = sample["t_s"] - origin
                writer.writerow([_cell(row.get(column)) for column in SAMPLE_COLUMNS])


_cuda_local = threading.local()


def cuda_mem_info():
    """(free_bytes, total_bytes) from the CUDA driver, or None if unavailable.

    Jetson has unified memory, so there is no separate VRAM pool: "used" is
    system memory the GPU driver accounts for and includes other processes.
    Only differences taken within one run are meaningful.
    """
    import ctypes as ct

    try:
        lib = getattr(_cuda_local, "lib", None)
        if lib is None:
            lib = ct.CDLL("libcuda.so.1")
            context = ct.c_void_p()
            if lib.cuInit(ct.c_uint(0)) or lib.cuDevicePrimaryCtxRetain(ct.byref(context), ct.c_int(0)):
                return None
            _cuda_local.lib = lib
            _cuda_local.context = context
        if lib.cuCtxSetCurrent(_cuda_local.context):
            return None
        free = ct.c_size_t()
        total = ct.c_size_t()
        if lib.cuMemGetInfo_v2(ct.byref(free), ct.byref(total)):
            return None
        return free.value, total.value
    except (OSError, AttributeError):
        return None


# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------

def _steady(frames, start, end):
    return [f for f in frames if start <= f["t"] <= end]


def _window_fps(frames, start, end, window=5.0):
    """Min/max FPS over consecutive ``window``-second buckets of the run."""
    buckets = int((end - start) // window)
    if buckets < 1:
        return None, None
    counts = [0] * buckets
    for f in frames:
        index = int((f["t"] - start) // window)
        if 0 <= index < buckets:
            counts[index] += 1
    rates = [c / window for c in counts]
    return round(min(rates), 3), round(max(rates), 3)


def _stage_report(frames):
    return {stage: distribution(f.get(stage) for f in frames) for stage in STAGES}


def _drift(frames, start, end):
    """First-vs-last window p95 latency and FPS (thermal / leak degradation)."""
    window = min(300.0, 0.2 * (end - start))
    if window < 1.0:
        return None
    first = [f for f in frames if f["t"] < start + window]
    last = [f for f in frames if f["t"] >= end - window]
    key = "e2e_ms" if any(f.get("e2e_ms") is not None for f in frames) else "total_ms"
    first_p95 = distribution(f.get(key) for f in first)
    last_p95 = distribution(f.get(key) for f in last)
    if not first_p95 or not last_p95 or not first_p95["p95"]:
        return None
    first_fps = len(first) / window
    last_fps = len(last) / window
    return {
        "window_s": round(window, 1),
        "latency_metric": key,
        "first_p95_ms": first_p95["p95"],
        "last_p95_ms": last_p95["p95"],
        "p95_change_pct": round((last_p95["p95"] - first_p95["p95"]) / first_p95["p95"] * 100.0, 2),
        "first_fps": round(first_fps, 3),
        "last_fps": round(last_fps, 3),
        "fps_change_pct": round((last_fps - first_fps) / first_fps * 100.0, 2) if first_fps else None,
    }


def _stability(frames):
    key = "e2e_ms" if any(f.get("e2e_ms") is not None for f in frames) else "total_ms"
    latency = distribution(f.get(key) for f in frames)
    result = {"latency_metric": key}
    if latency:
        result["latency_std_ms"] = latency["std"]
        result["latency_p99_minus_p50_ms"] = round(latency["p99"] - latency["p50"], 3)
        result["latency_cv"] = round(latency["std"] / latency["mean"], 4) if latency["mean"] else None
    raw = [f["jitter_raw_px"] for f in frames if f.get("jitter_raw_px") is not None]
    smooth = [f["jitter_smooth_px"] for f in frames if f.get("jitter_smooth_px") is not None]
    if raw and smooth:
        raw_mean = sum(raw) / len(raw)
        smooth_mean = sum(smooth) / len(smooth)
        result["keypoint_jitter_raw_px"] = round(raw_mean, 3)
        result["keypoint_jitter_smoothed_px"] = round(smooth_mean, 3)
        result["jitter_reduction_pct"] = round((1.0 - smooth_mean / raw_mean) * 100.0, 1) if raw_mean else None
    for name in ("head_forward", "torso_forward"):
        values = [f[name] for f in frames if f.get(name) is not None]
        if len(values) > 1:
            mean = sum(values) / len(values)
            result[name + "_std"] = round(math.sqrt(sum((v - mean) ** 2 for v in values) / len(values)), 4)
    states = [f["state"] for f in frames if f.get("state") not in (None, "UNCALIBRATED", "CALIBRATING")]
    flips = sum(1 for a, b in zip(states, states[1:]) if a != b)
    duration = frames[-1]["t"] - frames[0]["t"] if len(frames) > 1 else 0.0
    result["state_flips"] = flips
    result["state_flips_per_min"] = round(flips / (duration / 60.0), 2) if duration > 0 else None
    return result


def _recognition(frames, joint_names, confidence):
    total = len(frames)
    result = {
        "frames": total,
        "face_detect_rate": _rate(sum(1 for f in frames if f.get("face_found")), total),
        "person_lock_rate": _rate(sum(1 for f in frames if f.get("lock_ok")), total),
        "pose_valid_rate": _rate(sum(1 for f in frames if f.get("pose_valid")), total),
    }
    counts = {}
    for f in frames:
        state = f.get("state")
        if state:
            counts[state] = counts.get(state, 0) + 1
    result["state_distribution"] = {k: round(v / float(total), 4) for k, v in sorted(counts.items())} if total else {}
    joints = {}
    scored = [f["scores"] for f in frames if f.get("scores")]
    if joint_names and scored:
        for index, name in enumerate(joint_names):
            column = [s[index] for s in scored if index < len(s)]
            if column:
                joints[name] = {
                    "mean_score": round(sum(column) / len(column), 3),
                    "visible_rate": round(sum(1 for v in column if v >= confidence) / float(len(column)), 4),
                }
    result["joints"] = joints
    return result


def _memory(samples, recorder, extra):
    rss = [s["rss_mb"] for s in samples if s.get("rss_mb") is not None]
    ram = [s["ram_used_mb"] for s in samples if s.get("ram_used_mb") is not None]
    swap = [s["swap_used_mb"] for s in samples if s.get("swap_used_mb") is not None]
    _, hwm = read_process_memory()
    slope = None
    steady = [s for s in samples if s["phase"] == "steady" and s.get("rss_mb") is not None]
    if len(steady) > 2:
        slope = linear_slope([s["t_s"] / 60.0 for s in steady], [s["rss_mb"] for s in steady])
    gpu = recorder.gpu_mem_used_mb
    result = {
        "rss_peak_mb": round(max([hwm] if hwm else rss), 1) if (hwm or rss) else None,
        "rss_peak_sampled_mb": round(max(rss), 1) if rss else None,
        "rss_growth_mb_per_min": None if slope is None else round(slope, 3),
        "system_ram_peak_mb": round(max(ram), 1) if ram else None,
        "swap_peak_mb": round(max(swap), 1) if swap else None,
        "swap_growth_mb": round(swap[-1] - swap[0], 1) if len(swap) > 1 else None,
        "gpu_mem_used_peak_mb": round(max(gpu), 1) if gpu else None,
    }
    result.update(extra)
    if gpu and extra.get("gpu_mem_baseline_mb") is not None:
        result["gpu_mem_delta_mb"] = round(max(gpu) - extra["gpu_mem_baseline_mb"], 1)
    return result


def _energy(samples, frames, start, end, price_per_kwh):
    def mean_power(phase):
        values = [s["power_mw"] for s in samples if s["phase"] == phase and s.get("power_mw") is not None]
        return sum(values) / len(values) if values else None

    steady_power = [s["power_mw"] for s in samples if s["phase"] == "steady" and s.get("power_mw") is not None]
    if not steady_power:
        return {"available": False, "note": "no power source (tegrastats or INA3221 sysfs) on this machine"}
    duration = end - start
    mean_mw = sum(steady_power) / len(steady_power)
    idle_mw = mean_power("idle")
    energy_j = mean_mw / 1000.0 * duration
    count = len(frames)
    result = {
        "available": True,
        "duration_s": round(duration, 1),
        "power_mean_mw": round(mean_mw, 1),
        "power_max_mw": round(max(steady_power), 1),
        "power_idle_mw": None if idle_mw is None else round(idle_mw, 1),
        "energy_j": round(energy_j, 2),
        "energy_per_frame_mj": round(energy_j * 1000.0 / count, 2) if count else None,
        "energy_per_inference_mj": round(energy_j * 1000.0 / count, 2) if count else None,
    }
    if idle_mw is not None and count:
        net_j = max(0.0, mean_mw - idle_mw) / 1000.0 * duration
        result["net_energy_per_inference_mj"] = round(net_j * 1000.0 / count, 2)
    if price_per_kwh and count:
        result["cost_per_million_inferences"] = round(energy_j / count * 1e6 / 3.6e6 * price_per_kwh, 2)
    result["note"] = "board-reported input rail power; excludes wall-adapter and peripheral losses"
    return result


def summarize(recorder, sampler_samples, windows, options):
    """Build the run summary.

    ``windows``: dict with monotonic ``steady_start`` and ``steady_end``.
    ``options``: dict with ``confidence``, ``deadline_ms``, ``camera_fps``,
    ``temp_limit_c``, ``price_per_kwh``, ``gpu_extra`` (engine/baseline sizes).
    """
    start = windows["steady_start"]
    end = windows["steady_end"]
    model_only = options.get("mode") == "model"
    frames = _steady(recorder.frames, start, end)
    duration = max(1e-9, end - start)
    fps = len(frames) / duration
    stages = _stage_report(frames)
    key = "e2e_ms" if stages.get("e2e_ms") else "total_ms"
    latency = stages.get(key)
    window_min, window_max = _window_fps(frames, start, end)
    infer_total = stages.get("infer_ms")

    throughput = {
        "frames": len(frames),
        "duration_s": round(duration, 1),
        "fps_mean": round(fps, 3),
        "fps_window_min": window_min,
        "fps_window_max": window_max,
        "inference_rate_per_s": round(1000.0 / infer_total["mean"], 2) if infer_total and infer_total["mean"] else None,
    }

    reliability = {
        "deadline_ms": options["deadline_ms"],
        "deadline_metric": key,
        "deadline_miss_rate": _rate(
            sum(1 for f in frames if f.get(key) is not None and f[key] > options["deadline_ms"]),
            sum(1 for f in frames if f.get(key) is not None),
        ),
        "camera_timeouts": recorder.camera_timeouts,
        "errors": list(recorder.errors),
        "crashed": bool(recorder.errors),
    }
    pts = [f["pts_ns"] for f in frames if f.get("pts_ns") is not None]
    if len(pts) > 1 and options.get("camera_fps"):
        expected = (pts[-1] - pts[0]) / 1e9 * options["camera_fps"] + 1.0
        skipped = max(0.0, expected - len(pts))
        reliability["camera_frames_skipped"] = int(round(skipped))
        reliability["camera_frames_skipped_ratio"] = round(skipped / expected, 4) if expected else None
        reliability["skip_note"] = "latest-frame policy drops camera frames by design; gaps are not errors"
    temps = [s["temp_max_c"] for s in sampler_samples if s["phase"] == "steady" and s.get("temp_max_c") is not None]
    if temps:
        reliability["temp_max_c"] = round(max(temps), 1)
        reliability["temp_limit_c"] = options["temp_limit_c"]
        reliability["samples_over_temp_limit"] = sum(1 for t in temps if t >= options["temp_limit_c"])

    memory = _memory(sampler_samples, recorder, options.get("gpu_extra", {}))
    energy = _energy(sampler_samples, frames, start, end, options.get("price_per_kwh"))

    summary = {
        "latency_ms": stages,
        "throughput": throughput,
        "stability": _stability(frames),
        "drift": _drift(frames, start, end),
        "recognition": None if model_only else _recognition(frames, recorder.joint_names, options["confidence"]),
        "memory": memory,
        "energy": energy,
        "reliability": reliability,
    }
    summary["targets"] = check_targets(latency, fps, memory)
    return summary


def check_targets(latency, fps, memory):
    checks = []
    if latency:
        checks.append(_check("SYS-LAT-01", "latency p95 (ms)", latency["p95"], TARGET_P95_MS, latency["p95"] <= TARGET_P95_MS, "<="))
    checks.append(_check("SYS-THR-01", "FPS", round(fps, 3), TARGET_FPS, fps >= TARGET_FPS, ">="))
    if memory.get("rss_peak_mb") is not None:
        checks.append(_check("SYS-MEM-01", "RSS peak (MB)", memory["rss_peak_mb"], TARGET_RSS_MB, memory["rss_peak_mb"] <= TARGET_RSS_MB, "<="))
    return checks


def _check(check_id, metric, value, target, passed, operator):
    return {"id": check_id, "metric": metric, "value": value, "target": "%s %s" % (operator, target), "pass": bool(passed)}


def _fmt(value, unit=""):
    return "n/a" if value is None else "%s%s" % (value, unit)


def format_report(summary):
    """Human-readable console report of a summary dict."""
    lines = []
    add = lines.append
    stages = summary["latency_ms"]
    add("== 지연시간 (ms) ==")
    add("%-18s %8s %8s %8s %8s %8s" % ("단계", "mean", "p50", "p95", "p99", "max"))
    for name in STAGES:
        d = stages.get(name)
        if d:
            add("%-18s %8.2f %8.2f %8.2f %8.2f %8.2f" % (name, d["mean"], d["p50"], d["p95"], d["p99"], d["max"]))
    t = summary["throughput"]
    add("")
    add("== 처리량 ==")
    add("FPS %.2f (5초 구간 최소 %s / 최대 %s), 프레임 %d개, %.1fs" % (
        t["fps_mean"], _fmt(t["fps_window_min"]), _fmt(t["fps_window_max"]), t["frames"], t["duration_s"]))
    add("추론 처리율 %s/s (모델 단계만)" % _fmt(t["inference_rate_per_s"]))
    s = summary["stability"]
    add("")
    add("== 안정화 ==")
    add("지연 표준편차 %s ms, p99-p50 %s ms, CV %s" % (
        _fmt(s.get("latency_std_ms")), _fmt(s.get("latency_p99_minus_p50_ms")), _fmt(s.get("latency_cv"))))
    if "keypoint_jitter_raw_px" in s:
        add("키포인트 떨림 raw %.2fpx -> 스무딩 %.2fpx (%s%% 감소)" % (
            s["keypoint_jitter_raw_px"], s["keypoint_jitter_smoothed_px"], _fmt(s.get("jitter_reduction_pct"))))
    add("자세 상태 전환 %s회 (%s회/분)" % (s.get("state_flips"), _fmt(s.get("state_flips_per_min"))))
    drift = summary.get("drift")
    if drift:
        add("초반 %.0fs 대비 후반: p95 %+.1f%%, FPS %+.1f%%" % (
            drift["window_s"], drift["p95_change_pct"], drift["fps_change_pct"] or 0.0))
    r = summary["recognition"]
    if r:
        add("")
        add("== 인식율 ==")
        add("얼굴 검출 %s, 인물 잠금 %s, 유효 자세 %s" % (
            _fmt(r["face_detect_rate"]), _fmt(r["person_lock_rate"]), _fmt(r["pose_valid_rate"])))
        add("상태 분포 %s" % r["state_distribution"])
    m = summary["memory"]
    add("")
    add("== 메모리 ==")
    add("프로세스 RSS peak %s MB, 시스템 RAM peak %s MB, swap peak %s MB, RSS 증가 %s MB/분" % (
        _fmt(m["rss_peak_mb"]), _fmt(m["system_ram_peak_mb"]), _fmt(m["swap_peak_mb"]), _fmt(m["rss_growth_mb_per_min"])))
    add("GPU 메모리 peak %s MB (기준 대비 +%s MB), 엔진 디바이스 메모리 %s MB, 엔진 파일 %s MB" % (
        _fmt(m.get("gpu_mem_used_peak_mb")), _fmt(m.get("gpu_mem_delta_mb")),
        _fmt(m.get("engine_device_memory_mb")), _fmt(m.get("engine_file_mb"))))
    e = summary["energy"]
    add("")
    add("== 에너지 ==")
    if e["available"]:
        add("평균 %s mW (idle %s mW), 최대 %s mW, 총 %s J" % (
            e["power_mean_mw"], _fmt(e["power_idle_mw"]), e["power_max_mw"], e["energy_j"]))
        add("추론 1회당 %s mJ (idle 제외 %s mJ)" % (_fmt(e["energy_per_inference_mj"]), _fmt(e.get("net_energy_per_inference_mj"))))
        if "cost_per_million_inferences" in e:
            add("추론 100만 회 전력 비용 %s" % e["cost_per_million_inferences"])
    else:
        add(e["note"])
    rel = summary["reliability"]
    add("")
    add("== 안전성 ==")
    add("데드라인 %.0fms 초과율 %s, 카메라 타임아웃 %d회, 오류 %d건" % (
        rel["deadline_ms"], _fmt(rel["deadline_miss_rate"]), rel["camera_timeouts"], len(rel["errors"])))
    if "temp_max_c" in rel:
        add("최고 온도 %.1fC (한계 %.0fC 초과 샘플 %d개)" % (rel["temp_max_c"], rel["temp_limit_c"], rel["samples_over_temp_limit"]))
    for error in rel["errors"]:
        add("  오류: %s" % error)
    add("")
    add("== 검증 계획 합격 기준 ==")
    for check in summary["targets"]:
        add("[%s] %s %s = %s (목표 %s)" % ("PASS" if check["pass"] else "FAIL", check["id"], check["metric"], check["value"], check["target"]))
    return "\n".join(lines)


COMPARISON_COLUMNS = (
    "run_id",
    "label",
    "fps_mean",
    "latency_p50_ms",
    "latency_p95_ms",
    "latency_p99_ms",
    "latency_std_ms",
    "face_detect_rate",
    "pose_valid_rate",
    "rss_peak_mb",
    "gpu_mem_delta_mb",
    "power_mean_mw",
    "energy_per_inference_mj",
    "temp_max_c",
    "deadline_miss_rate",
    "p95_change_pct",
)


def flatten_summary(run_id, label, summary):
    """One comparison.csv row from a ``summary.json`` dict."""
    stages = summary["latency_ms"]
    latency = stages.get("e2e_ms") or stages.get("total_ms") or {}
    drift = summary.get("drift") or {}
    recognition = summary.get("recognition") or {}
    return {
        "run_id": run_id,
        "label": label,
        "fps_mean": summary["throughput"]["fps_mean"],
        "latency_p50_ms": latency.get("p50"),
        "latency_p95_ms": latency.get("p95"),
        "latency_p99_ms": latency.get("p99"),
        "latency_std_ms": latency.get("std"),
        "face_detect_rate": recognition.get("face_detect_rate"),
        "pose_valid_rate": recognition.get("pose_valid_rate"),
        "rss_peak_mb": summary["memory"].get("rss_peak_mb"),
        "gpu_mem_delta_mb": summary["memory"].get("gpu_mem_delta_mb"),
        "power_mean_mw": summary["energy"].get("power_mean_mw"),
        "energy_per_inference_mj": summary["energy"].get("energy_per_inference_mj"),
        "temp_max_c": summary["reliability"].get("temp_max_c"),
        "deadline_miss_rate": summary["reliability"].get("deadline_miss_rate"),
        "p95_change_pct": drift.get("p95_change_pct"),
    }
