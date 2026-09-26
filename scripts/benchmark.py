#!/usr/bin/env python3
"""Benchmark the Jetson posture pipeline.

  run      full pipeline (camera -> face -> TensorRT -> posture -> JPEG)
  model    TensorRT engine only, synthetic frame, no camera
  rtt      HTTP round trip against a running web_mvp/server.py
  compare  merge runs/*/summary.json into benchmarks/comparison.csv

Run on the Jetson with ``./scripts/nano_python.sh scripts/benchmark.py run``.
Results go to benchmarks/runs/<run-id>/ (see benchmarks/README.md).
"""

from __future__ import print_function

import argparse
import csv
import glob
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import bench

RUNS_DIR = ROOT / "benchmarks" / "runs"


# --------------------------------------------------------------------------
# Environment metadata (docs/system_validation_plan.md section 2)
# --------------------------------------------------------------------------

def _output(command):
    try:
        return subprocess.check_output(command, cwd=str(ROOT), stderr=subprocess.DEVNULL).decode().strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def environment(args):
    info = {
        "git_sha": _output(["git", "rev-parse", "HEAD"]),
        "git_dirty": bool(_output(["git", "status", "--porcelain", "--untracked-files=no"])),
        "date": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "kernel": platform.release(),
        "nvpmodel": _output(["nvpmodel", "-q"]),
        "l4t": None,
        "power_source": None,
        "args": {k: v for k, v in sorted(vars(args).items()) if k != "func"},
    }
    try:
        with open("/etc/nv_tegra_release") as stream:
            info["l4t"] = stream.readline().strip()
    except (IOError, OSError):
        pass
    try:
        import tensorrt

        info["tensorrt"] = tensorrt.__version__
    except ImportError:
        info["tensorrt"] = None
    engine = getattr(args, "engine", None)
    if engine and os.path.isfile(engine):
        info["engine_sha256"] = _sha256(engine)
        info["engine_file_mb"] = round(os.path.getsize(engine) / 1048576.0, 2)
    baseline = getattr(args, "baseline", None)
    if baseline and os.path.isfile(baseline):
        info["baseline_sha256"] = _sha256(baseline)
    return info


def make_run_dir(label, suffix=""):
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", label).strip("-") or "run"
    run_id = "%s-%s%s" % (time.strftime("%Y%m%d-%H%M%S"), safe, suffix)
    path = RUNS_DIR / run_id
    path.mkdir(parents=True)
    return run_id, path


def write_json(path, payload):
    with open(str(path), "w") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, ensure_ascii=False)
        stream.write("\n")


def wait_until(deadline, is_alive=lambda: True):
    """Sleep to ``deadline`` (monotonic); False if ``is_alive`` failed first."""
    while time.monotonic() < deadline:
        if not is_alive():
            return False
        time.sleep(0.2)
    return True


def gpu_used_mb():
    info = bench.cuda_mem_info()
    return None if info is None else (info[1] - info[0]) / 1048576.0


def summary_options(args, gpu_extra, mode):
    return {
        "mode": mode,
        "confidence": args.confidence,
        "deadline_ms": args.deadline_ms,
        "camera_fps": getattr(args, "camera_fps", None),
        "temp_limit_c": args.temp_limit,
        "price_per_kwh": args.price_per_kwh,
        "gpu_extra": gpu_extra,
    }


def finish(args, run_id, run_dir, recorder, sampler, windows, gpu_extra, mode, config):
    summary = bench.summarize(recorder, sampler.samples, windows, summary_options(args, gpu_extra, mode))
    summary["mode"] = mode
    summary["power_source"] = sampler.power_source
    config["power_source"] = sampler.power_source
    write_json(run_dir / "config.json", config)
    write_json(run_dir / "summary.json", summary)
    recorder.write_frames_csv(str(run_dir / "frames.csv"))
    sampler.write_csv(str(run_dir / "system.csv"), recorder.origin)
    print(bench.format_report(summary))
    print("\n결과 저장: %s" % run_dir)
    return 0 if not recorder.errors and windows["steady_end"] > windows["steady_start"] else 1


# --------------------------------------------------------------------------
# run: full pipeline
# --------------------------------------------------------------------------

def cmd_run(args):
    from web_mvp import server

    if not os.path.isfile(args.engine):
        print("ERROR: TensorRT engine not found: %s" % args.engine, file=sys.stderr)
        return 2
    if not os.path.isfile(args.baseline):
        print("WARNING: no baseline at %s; posture classification is skipped (timings still valid)." % args.baseline, file=sys.stderr)

    run_id, run_dir = make_run_dir(args.label)
    config = environment(args)
    config["run_id"] = run_id
    recorder = bench.FrameRecorder(args.confidence)
    recorder.gpu_mem_probe = bench.cuda_mem_info
    gpu_baseline = gpu_used_mb()

    sampler = bench.SystemSampler(args.sample_interval, args.power_file)
    sampler.start()
    state = server.State()
    http_server = None
    worker = threading.Thread(target=server.worker, args=(state, args, recorder))
    worker.daemon = True

    windows = {"steady_start": 0.0, "steady_end": 0.0}
    try:
        if args.idle_seconds > 0:
            print("idle 전력 측정 %.0fs (카메라·추론 정지)..." % args.idle_seconds)
            sampler.mark("idle")
            time.sleep(args.idle_seconds)
        sampler.mark("startup")
        worker.start()
        if args.serve_port:
            http_server = start_http(server, state, args.serve_port)
        deadline = time.monotonic() + args.startup_timeout
        while not recorder.frames and worker.is_alive() and time.monotonic() < deadline:
            time.sleep(0.1)
        if not recorder.frames:
            print("ERROR: no frame was processed. %s" % ("; ".join(recorder.errors) or "startup timeout"), file=sys.stderr)
            return 1

        print("warm-up %.0fs..." % args.warmup)
        sampler.mark("warmup")
        wait_until(time.monotonic() + args.warmup, worker.is_alive)
        windows["steady_start"] = time.monotonic()
        sampler.mark("steady")
        print("측정 %.0fs..." % args.duration)
        wait_until(windows["steady_start"] + args.duration, worker.is_alive)
        windows["steady_end"] = time.monotonic()
    except KeyboardInterrupt:
        print("중단됨; 지금까지의 데이터로 요약합니다.")
        windows["steady_end"] = time.monotonic()
    finally:
        sampler.mark("done")
        state.running = False
        if http_server is not None:
            http_server.shutdown()
            http_server.server_close()
        worker.join(timeout=10.0)
        sampler.stop()

    if windows["steady_start"] == 0.0:
        windows["steady_start"] = recorder.frames[0]["t"] if recorder.frames else windows["steady_end"]
    gpu_extra = {"gpu_mem_baseline_mb": gpu_baseline, "engine_file_mb": config.get("engine_file_mb")}
    return finish(args, run_id, run_dir, recorder, sampler, windows, gpu_extra, "run", config)


def start_http(server, state, port):
    """Serve the dashboard from this process so an ``rtt`` client can measure it."""
    server.Handler.state = state
    server.Handler.index_html = (ROOT / "web" / "index.html").read_text()
    httpd = server.ThreadingHTTPServer(("0.0.0.0", port), server.Handler)
    thread = threading.Thread(target=httpd.serve_forever)
    thread.daemon = True
    thread.start()
    print("HTTP 서버 :%d (rtt 측정용)" % port)
    return httpd


# --------------------------------------------------------------------------
# model: engine only
# --------------------------------------------------------------------------

def cmd_model(args):
    import numpy as np
    from web_mvp import server

    if not os.path.isfile(args.engine):
        print("ERROR: TensorRT engine not found: %s" % args.engine, file=sys.stderr)
        return 2

    run_id, run_dir = make_run_dir(args.label, "-model")
    config = environment(args)
    config["run_id"] = run_id
    recorder = bench.FrameRecorder(args.confidence)
    recorder.gpu_mem_probe = bench.cuda_mem_info

    if args.image:
        import cv2

        frame = cv2.imread(args.image)
        if frame is None:
            print("ERROR: cannot read image: %s" % args.image, file=sys.stderr)
            return 2
    else:
        frame = np.random.RandomState(0).randint(0, 256, (720, 1280, 3)).astype(np.uint8)

    sampler = bench.SystemSampler(args.sample_interval, args.power_file)
    sampler.start()
    windows = {"steady_start": 0.0, "steady_end": 0.0}
    gpu_baseline = gpu_used_mb()
    gpu_extra = {"gpu_mem_baseline_mb": gpu_baseline}
    device = None
    try:
        if args.idle_seconds > 0:
            print("idle 전력 측정 %.0fs..." % args.idle_seconds)
            sampler.mark("idle")
            time.sleep(args.idle_seconds)
        sampler.mark("startup")
        _, engine, context, device, arrays, pointers, input_i, outputs, heatmap_i = server.load_engine(args.engine)
        gpu_extra = {
            "gpu_mem_baseline_mb": gpu_baseline,
            "engine_file_mb": config.get("engine_file_mb"),
            "engine_device_memory_mb": round(engine.device_memory_size / 1048576.0, 2),
        }
        recorder.probe_gpu_memory()

        def one_inference():
            timing = {}
            began = time.perf_counter()
            server.infer(frame, context, device, arrays, pointers, input_i, outputs, heatmap_i, timing)
            total = (time.perf_counter() - began) * 1000.0
            record = {"t": time.monotonic(), "infer_ms": total, "total_ms": total}
            record.update(timing)
            recorder.frame(record)

        print("warm-up %.0fs..." % args.warmup)
        sampler.mark("warmup")
        end_warmup = time.monotonic() + args.warmup
        while time.monotonic() < end_warmup:
            one_inference()
        windows["steady_start"] = time.monotonic()
        sampler.mark("steady")
        print("측정 %.0fs..." % args.duration)
        end = windows["steady_start"] + args.duration
        count = 0
        while time.monotonic() < end:
            one_inference()
            count += 1
            if count % 30 == 0:
                recorder.probe_gpu_memory()
        windows["steady_end"] = time.monotonic()
    except KeyboardInterrupt:
        windows["steady_end"] = time.monotonic()
    except Exception as error:
        recorder.error(error)
        windows["steady_end"] = time.monotonic()
    finally:
        sampler.mark("done")
        sampler.stop()
        if device is not None:
            device.close()
    if windows["steady_start"] == 0.0:
        windows["steady_start"] = windows["steady_end"]
    return finish(args, run_id, run_dir, recorder, sampler, windows, gpu_extra, "model", config)


# --------------------------------------------------------------------------
# rtt: HTTP round trip
# --------------------------------------------------------------------------

def cmd_rtt(args):
    from urllib.error import URLError
    from urllib.request import urlopen

    url = args.url.rstrip("/") + "/api/pose"
    interval = 1.0 / args.rate
    samples = []
    errors = 0
    end = time.monotonic() + args.duration
    print("RTT 측정 %s (%.0fHz, %.0fs)..." % (url, args.rate, args.duration))
    while time.monotonic() < end:
        began = time.monotonic()
        sent = time.time()
        counter = time.perf_counter()
        try:
            body = urlopen(url, timeout=5.0).read()
            rtt_ms = (time.perf_counter() - counter) * 1000.0
            received = time.time()
            pose = json.loads(body.decode("utf-8"))
        except (URLError, OSError, ValueError):
            errors += 1
        else:
            samples.append((sent, received, rtt_ms, pose))
        time.sleep(max(0.0, interval - (time.monotonic() - began)))

    if not samples:
        print("ERROR: no successful response from %s" % url, file=sys.stderr)
        return 1

    # Server clock offset from the request midpoint (error <= RTT/2), median
    # over all polls; needed because the client is usually another machine.
    offsets = sorted(p["server_time"] - (s + r) / 2.0 for s, r, _, p in samples if "server_time" in p)
    offset = offsets[len(offsets) // 2] if offsets else 0.0
    seen = set()
    age_ms, first_ms = [], []
    for _, received, _, pose in samples:
        if pose.get("captured_at") is None:
            continue
        age = (received + offset - pose["captured_at"]) * 1000.0
        age_ms.append(age)
        frame_id = pose.get("frame_id")
        if frame_id not in seen:
            seen.add(frame_id)
            first_ms.append(age)
    frame_ids = [p.get("frame_id") for _, _, _, p in samples if p.get("frame_id") is not None]
    span = samples[-1][1] - samples[0][0]
    summary = {
        "url": url,
        "polls": len(samples),
        "errors": errors,
        "http_rtt_ms": bench.distribution(r for _, _, r, _ in samples),
        "result_age_ms": bench.distribution(age_ms),
        "capture_to_client_first_seen_ms": bench.distribution(first_ms),
        "server_fps_observed": round((max(frame_ids) - min(frame_ids)) / span, 3) if len(frame_ids) > 1 and span > 0 else None,
        "clock_offset_ms": round(offset * 1000.0, 2),
        "note": "http_rtt_ms = request->response. result_age_ms = camera capture -> client received, "
        "sampled every poll (includes waiting for the next frame). first_seen_ms only counts each frame's first poll, "
        "so it is bounded below by the poll interval. Server/client clocks aligned by request midpoint (+/- RTT/2).",
    }
    run_id, run_dir = make_run_dir(args.label, "-rtt")
    write_json(run_dir / "summary.json", summary)
    write_json(run_dir / "config.json", {"run_id": run_id, "date": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "args": {k: v for k, v in vars(args).items() if k != "func"}})
    for name in ("http_rtt_ms", "result_age_ms", "capture_to_client_first_seen_ms"):
        d = summary[name]
        if d:
            print("%-34s mean %8.2f  p50 %8.2f  p95 %8.2f  p99 %8.2f  max %8.2f" % (name, d["mean"], d["p50"], d["p95"], d["p99"], d["max"]))
    print("오류 %d건, 서버 FPS(관측) %s, 시계 오프셋 %.1f ms" % (errors, summary["server_fps_observed"], summary["clock_offset_ms"]))
    print("\n결과 저장: %s" % run_dir)
    return 0


# --------------------------------------------------------------------------
# compare
# --------------------------------------------------------------------------

def cmd_compare(args):
    paths = args.runs or sorted(glob.glob(str(RUNS_DIR / "*")))
    rows = []
    for path in paths:
        summary_path = os.path.join(path, "summary.json")
        if not os.path.isfile(summary_path):
            continue
        with open(summary_path) as stream:
            summary = json.load(stream)
        if "latency_ms" not in summary:
            continue  # rtt run
        config = {}
        if os.path.isfile(os.path.join(path, "config.json")):
            with open(os.path.join(path, "config.json")) as stream:
                config = json.load(stream)
        label = (config.get("args") or {}).get("label", "")
        rows.append(bench.flatten_summary(os.path.basename(path.rstrip("/")), label, summary))
    if not rows:
        print("no runs found", file=sys.stderr)
        return 1
    with open(args.output, "w") as stream:
        writer = csv.DictWriter(stream, fieldnames=bench.COMPARISON_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print("%d runs -> %s" % (len(rows), args.output))
    return 0


# --------------------------------------------------------------------------

def add_common(parser):
    parser.add_argument("--label", default="run", help="name appended to the run id (e.g. fp16-fps8)")
    parser.add_argument("--warmup", type=float, default=15.0, help="seconds excluded from statistics (plan: 180)")
    parser.add_argument("--duration", type=float, default=60.0, help="measured seconds (plan: 1800)")
    parser.add_argument("--idle-seconds", type=float, default=10.0, help="idle power baseline before the run; 0 skips")
    parser.add_argument("--sample-interval", type=float, default=1.0)
    parser.add_argument("--power-file", default=None, help="sysfs power file in mW if tegrastats is unavailable")
    parser.add_argument("--price-per-kwh", type=float, default=0.0, help="electricity price; enables cost per 1M inferences")
    parser.add_argument("--deadline-ms", type=float, default=200.0, help="per-frame latency budget (plan: 200)")
    parser.add_argument("--temp-limit", type=float, default=80.0, help="C; samples at or above are counted")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--engine", default=str(ROOT / "models" / "pose-resnet18-fp16.engine"))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command")
    sub.required = True

    run = sub.add_parser("run", help="full pipeline benchmark with the CSI camera")
    add_common(run)
    run.add_argument("--baseline", default=str(ROOT / "data" / "posture_baseline.json"))
    run.add_argument("--events", default=os.devnull, help="alert log (discarded by default)")
    run.add_argument("--camera-fps", type=int, default=30)
    run.add_argument("--head-threshold", type=float, default=0.14)
    run.add_argument("--torso-threshold", type=float, default=0.12)
    run.add_argument("--hold-seconds", type=float, default=3.0)
    run.add_argument("--warmup-seconds", type=float, default=3.0, help="(worker) calibration countdown; unused here")
    run.add_argument("--calibration-seconds", type=float, default=5.0, help="(worker) unused here")
    run.add_argument("--startup-timeout", type=float, default=90.0)
    run.add_argument("--serve-port", type=int, default=0, help="also serve HTTP on this port so `rtt` can run against the loaded system")
    run.set_defaults(func=cmd_run)

    model = sub.add_parser("model", help="engine-only speed/memory/energy, no camera")
    add_common(model)
    model.add_argument("--image", default=None, help="use this image instead of a synthetic frame")
    model.set_defaults(func=cmd_model)

    rtt = sub.add_parser("rtt", help="HTTP round trip to a running server")
    rtt.add_argument("--url", default="http://127.0.0.1:8080")
    rtt.add_argument("--rate", type=float, default=20.0, help="polls per second")
    rtt.add_argument("--duration", type=float, default=30.0)
    rtt.add_argument("--label", default="run")
    rtt.set_defaults(func=cmd_rtt)

    compare = sub.add_parser("compare", help="write benchmarks/comparison.csv")
    compare.add_argument("runs", nargs="*", help="run directories (default: all)")
    compare.add_argument("--output", default=str(ROOT / "benchmarks" / "comparison.csv"))
    compare.set_defaults(func=cmd_compare)

    args = parser.parse_args()
    if getattr(args, "duration", 1) <= 0:
        parser.error("--duration must be positive")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
