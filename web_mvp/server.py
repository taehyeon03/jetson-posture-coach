"""On-device posture web MVP for Jetson Nano.

The camera, TensorRT inference, JPEG encoding, and HTTP server all run on the
Jetson. The browser only receives the latest JPEG and pose JSON. This is an
MVP transport: MJPEG is intentionally replaceable with WebRTC later.
"""
import argparse
import collections
import ctypes as ct
import json
import os
import signal
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from socketserver import ThreadingMixIn
from urllib.parse import urlparse

import cv2
import numpy as np
import tensorrt as trt

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.camera import CsiCamera
from src.posture import (
    Baseline,
    KeypointSmoother,
    PersonLock,
    SteadyMetrics,
    SustainedAlert,
    classify,
    diagnose_pose,
    extract_metrics,
)

MISSING_LABELS_KO = {"nose": "코", "ear": "귀", "shoulder": "어깨", "hip": "엉덩이"}


def missing_message(point_map, confidence, prefix="안 보여요"):
    """Human-readable message naming the exact landmark(s) that are too
    low-confidence right now, e.g. "안 보여요: 엉덩이" instead of always
    blaming head/shoulder/hip together."""
    diagnosis = diagnose_pose(point_map, confidence)
    if not diagnosis["missing"]:
        return None
    names = "·".join(MISSING_LABELS_KO[part] for part in diagnosis["missing"])
    return "{}: {} (카메라 쪽으로 더 옆으로 돌아 앉아보세요)".format(prefix, names)


KEYPOINTS = json.loads(
    (Path(__file__).resolve().parent.parent / "models/Pose-ResNet18-Body/human_pose.json").read_text()
)["keypoints"]
# Once a person is locked, inference runs on a crop around their last known
# bounding box instead of the full frame. This stops the single-person
# heatmap model from snapping to a different person when several are visible.
ROI_MARGIN_RATIO = 0.6
ROI_MIN_VALID_POINTS = 4
ROI_MIN_SIZE = 40
ROI_LOST_RESET_FRAMES = 10

EDGES = [
    ("left_ear", "left_eye"), ("left_eye", "nose"),
    ("nose", "right_eye"), ("right_eye", "right_ear"),
    ("left_ear", "left_shoulder"), ("right_ear", "right_shoulder"),
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"), ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"), ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
]


class Device:
    def __init__(self):
        self.lib = ct.CDLL("libcuda.so.1")
        self.context = ct.c_void_p()
        self.pointers = []
        self.call("cuInit", ct.c_uint(0))
        self.call("cuDevicePrimaryCtxRetain", ct.byref(self.context), ct.c_int(0))
        self.call("cuCtxSetCurrent", self.context)

    def call(self, name, *args):
        result = getattr(self.lib, name)(*args)
        if result:
            raise RuntimeError("{}: CUDA error {}".format(name, result))

    def alloc(self, array):
        pointer = ct.c_uint64()
        self.call("cuMemAlloc_v2", ct.byref(pointer), ct.c_size_t(array.nbytes))
        self.pointers.append(pointer)
        return pointer

    def close(self):
        for pointer in self.pointers:
            self.call("cuMemFree_v2", pointer)
        self.call("cuDevicePrimaryCtxRelease", ct.c_int(0))


class State:
    def __init__(self):
        self.condition = threading.Condition()
        self.jpeg = None
        self.pose = {
            "status": "starting",
            "state": "OFFLINE",
            "message": "AI와 카메라를 시작하고 있습니다.",
            "points": [],
            "fps": 0.0,
            "latency_ms": 0.0,
            "calibrated": False,
            "calibration_phase": "idle",
            "calibration_remaining": 0.0,
            "head_delta": None,
            "torso_delta": None,
            "confidence": None,
            "side": None,
            "bad_for": 0.0,
            "alert_count": 0,
            "events": [],
        }
        self.frame_id = 0
        self.running = True
        self.calibration_requested = False
        self.events = collections.deque(maxlen=20)
        self.alert_count = 0
        self.started_at = time.monotonic()

    def publish(self, jpeg, pose):
        with self.condition:
            self.jpeg = jpeg
            self.pose = pose
            self.frame_id += 1
            self.condition.notify_all()

    def snapshot(self):
        with self.condition:
            return self.frame_id, self.jpeg, dict(self.pose)

    def request_calibration(self):
        with self.condition:
            phase = self.pose.get("calibration_phase", "idle")
            if phase != "idle" or self.calibration_requested:
                return False, "이미 보정 중입니다."
            self.calibration_requested = True
            return True, "보정을 시작합니다. 바른 자세로 앉아 주세요."

    def take_calibration_request(self):
        with self.condition:
            requested = self.calibration_requested
            self.calibration_requested = False
            return requested

    def add_event(self, state, head_delta, torso_delta, event_path):
        event = {
            "time": time.strftime("%H:%M:%S"),
            "state": state,
            "head_delta": None if head_delta is None else round(head_delta, 3),
            "torso_delta": None if torso_delta is None else round(torso_delta, 3),
        }
        with self.condition:
            self.events.appendleft(event)
            self.alert_count += 1
        parent = os.path.dirname(event_path)
        if parent and not os.path.isdir(parent):
            os.makedirs(parent)
        stored = dict(event)
        stored["time"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        with open(event_path, "a") as stream:
            stream.write(json.dumps(stored, sort_keys=True) + "\n")


def load_engine(engine_path):
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(Path(engine_path).read_bytes())
    if engine is None:
        raise RuntimeError("cannot deserialize TensorRT engine: {}".format(engine_path))
    context = engine.create_execution_context()
    device = Device()
    arrays, pointers, inputs, outputs, bindings = [], [], [], [], []
    for i in range(engine.num_bindings):
        shape = tuple(engine.get_binding_shape(i))
        if any(n <= 0 for n in shape):
            raise RuntimeError("dynamic bindings are not supported: {}".format(shape))
        array = np.empty(shape, dtype=trt.nptype(engine.get_binding_dtype(i)))
        arrays.append(array)
        pointers.append(device.alloc(array))
        (inputs if engine.binding_is_input(i) else outputs).append(i)
        bindings.append({"name": engine.get_binding_name(i), "shape": shape})
    if len(inputs) != 1:
        raise RuntimeError("expected one TensorRT image input")
    heatmap_indices = [
        i for i in outputs if arrays[i].ndim == 4 and arrays[i].shape[1] == len(KEYPOINTS)
    ]
    if len(heatmap_indices) != 1:
        raise RuntimeError("cannot identify the pose heatmap output")
    return runtime, engine, context, device, arrays, pointers, inputs[0], outputs, heatmap_indices[0]


def infer(frame, context, device, arrays, pointers, input_i, outputs, heatmap_i):
    _, _, height, width = arrays[input_i].shape
    rgb = cv2.cvtColor(cv2.resize(frame, (width, height)), cv2.COLOR_BGR2RGB)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    tensor = ((rgb.astype(np.float32) / 255.0 - mean) / std).transpose(2, 0, 1)[None]
    arrays[input_i][...] = tensor
    device.call("cuMemcpyHtoD_v2", pointers[input_i], ct.c_void_p(arrays[input_i].ctypes.data), ct.c_size_t(arrays[input_i].nbytes))
    if not context.execute_v2([p.value for p in pointers]):
        raise RuntimeError("TensorRT inference failed")
    for i in outputs:
        device.call("cuMemcpyDtoH_v2", ct.c_void_p(arrays[i].ctypes.data), pointers[i], ct.c_size_t(arrays[i].nbytes))
    heatmaps = arrays[heatmap_i][0]
    points = []
    for name, heatmap in zip(KEYPOINTS, heatmaps):
        y, x = np.unravel_index(heatmap.argmax(), heatmap.shape)
        points.append({
            "name": name,
            "x": (float(x) + 0.5) / heatmap.shape[1],
            "y": (float(y) + 0.5) / heatmap.shape[0],
            "score": float(heatmap[y, x]),
        })
    return points


def bbox_from_points(points, frame_w, frame_h, confidence, margin_ratio=ROI_MARGIN_RATIO):
    """Bounding box (in full-frame pixels) around the locked person's joints."""
    valid = [(p["x"] * frame_w, p["y"] * frame_h) for p in points if p["score"] >= confidence]
    if len(valid) < ROI_MIN_VALID_POINTS:
        return None
    xs = [v[0] for v in valid]
    ys = [v[1] for v in valid]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    margin_x = (x1 - x0) * margin_ratio
    margin_y = (y1 - y0) * margin_ratio
    x0 = max(0, int(x0 - margin_x))
    y0 = max(0, int(y0 - margin_y))
    x1 = min(frame_w, int(x1 + margin_x))
    y1 = min(frame_h, int(y1 + margin_y))
    if x1 - x0 < ROI_MIN_SIZE or y1 - y0 < ROI_MIN_SIZE:
        return None
    return (x0, y0, x1, y1)


def infer_tracked(frame, roi, context, device, arrays, pointers, input_i, outputs, heatmap_i):
    """Run ``infer`` on ``roi`` (or the full frame) and map points back to
    full-frame fractions, so downstream code never needs to know a crop
    happened."""
    frame_h, frame_w = frame.shape[:2]
    if roi is None:
        crop = frame
        offset_x, offset_y = 0, 0
        crop_w, crop_h = frame_w, frame_h
    else:
        x0, y0, x1, y1 = roi
        crop = frame[y0:y1, x0:x1]
        offset_x, offset_y = x0, y0
        crop_h, crop_w = crop.shape[:2]

    raw_points = infer(crop, context, device, arrays, pointers, input_i, outputs, heatmap_i)
    if roi is None:
        return raw_points
    return [
        {
            "name": p["name"],
            "x": (offset_x + p["x"] * crop_w) / frame_w,
            "y": (offset_y + p["y"] * crop_h) / frame_h,
            "score": p["score"],
        }
        for p in raw_points
    ]


def worker(state, args):
    device = None
    camera = None
    camera_open = False
    try:
        resources = load_engine(args.engine)
        _, _, context, device, arrays, pointers, input_i, outputs, heatmap_i = resources
        camera = CsiCamera(fps=args.camera_fps)
        baseline = Baseline.load(args.baseline) if Path(args.baseline).is_file() else None
        alert = SustainedAlert(args.hold_seconds)
        smoother = KeypointSmoother()
        steady = SteadyMetrics()
        person_lock = PersonLock()
        calibration_phase = "idle"
        calibration_started = 0.0
        calibration_samples = []
        notice_message = None
        notice_until = 0.0
        previous = time.monotonic()
        measured = []
        locked_bbox = None
        lost_streak = 0

        camera.__enter__()
        camera_open = True
        while state.running:
            frame = camera.read()
            if frame is None:
                continue
            started = time.monotonic()
            points = infer_tracked(frame, locked_bbox, context, device, arrays, pointers, input_i, outputs, heatmap_i)
            frame_h, frame_w = frame.shape[:2]
            point_map_raw = {
                p["name"]: (p["x"] * frame_w, p["y"] * frame_h, p["score"])
                for p in points
            }
            # Gate on eye position before trusting this frame's body at all:
            # a bystander's face landing in the crop must not relock the ROI
            # or feed the smoother/metrics with someone else's posture.
            is_locked_person = person_lock.update(point_map_raw, args.confidence)

            if is_locked_person:
                new_bbox = bbox_from_points(points, frame_w, frame_h, args.confidence)
                if new_bbox is not None:
                    locked_bbox = new_bbox
                    lost_streak = 0
                else:
                    lost_streak += 1
                    if lost_streak >= ROI_LOST_RESET_FRAMES:
                        locked_bbox = None
                point_map = smoother.smooth(point_map_raw, started)
                metrics = extract_metrics(point_map, args.confidence)
                # Send the smoothed positions to the browser too, so the
                # joint overlay itself stops twitching frame to frame.
                points = [
                    {"name": name, "x": x / frame_w, "y": y / frame_h, "score": score}
                    for name, (x, y, score) in point_map.items()
                ]
            else:
                # Wrong person (or nobody verifiable) in view this frame:
                # drop the crop lock so the next frame searches the full
                # frame again, and don't let this frame's points count.
                locked_bbox = None
                lost_streak = 0
                point_map = point_map_raw
                metrics = None

            # Bridges a keypoint (usually the hip) dropping below threshold
            # for a frame or two, instead of the display flickering to
            # NO_POSE on every transient occlusion.
            steady_metrics = steady.update(metrics, started)

            now = time.monotonic()
            measured.append(now - previous)
            measured = measured[-30:]
            previous = now
            fps = 1.0 / (sum(measured) / len(measured)) if measured else 0.0
            latency_ms = (now - started) * 1000.0

            if state.take_calibration_request():
                calibration_phase = "countdown"
                calibration_started = now
                calibration_samples = []
                alert.update("GOOD", now)

            posture_state = "UNCALIBRATED"
            message = "먼저 바른 자세 기준을 보정하세요."
            head_delta = None
            torso_delta = None
            bad_for = 0.0
            calibration_remaining = 0.0

            if calibration_phase == "countdown":
                calibration_remaining = max(0.0, args.warmup_seconds - (now - calibration_started))
                posture_state = "CALIBRATING"
                message = missing_message(point_map, args.confidence) or "바르게 앉아 주세요. 곧 기준 자세를 수집합니다."
                if calibration_remaining <= 0:
                    calibration_phase = "collecting"
                    calibration_started = now
                    calibration_remaining = args.calibration_seconds

            if calibration_phase == "collecting":
                if metrics is not None:
                    calibration_samples.append(metrics)
                calibration_remaining = max(0.0, args.calibration_seconds - (now - calibration_started))
                posture_state = "CALIBRATING"
                message = missing_message(point_map, args.confidence) or "움직이지 말고 바른 자세를 유지하세요."
                if calibration_remaining <= 0:
                    minimum = max(8, int(args.calibration_seconds * max(1.0, fps) * 0.35))
                    if len(calibration_samples) >= minimum:
                        baseline = Baseline.from_samples(calibration_samples)
                        baseline.save(args.baseline)
                        posture_state = "GOOD"
                        message = "개인 기준 저장 완료: 유효 프레임 {}개".format(baseline.samples)
                    else:
                        posture_state = "NO_POSE"
                        message = missing_message(point_map, args.confidence, prefix="보정 실패 - 안 보여요") or "보정 실패: 머리·어깨·엉덩이가 모두 보이게 조정하세요."
                    notice_message = message
                    notice_until = now + 4.0
                    calibration_phase = "idle"
                    calibration_remaining = 0.0

            elif calibration_phase == "idle" and baseline is not None:
                posture_state, head_delta, torso_delta = classify(
                    steady_metrics,
                    baseline,
                    args.head_threshold,
                    args.torso_threshold,
                )
                fired, bad_for = alert.update(posture_state, now)
                if posture_state == "NO_POSE":
                    message = missing_message(point_map, args.confidence) or "머리·어깨·엉덩이가 보이도록 앉아 주세요."
                elif posture_state == "GOOD":
                    message = "바른 자세입니다."
                else:
                    message = "자세를 바로잡아 주세요."
                if fired:
                    state.add_event(posture_state, head_delta, torso_delta, args.events)

            if now < notice_until and notice_message:
                message = notice_message

            # Keep inference at camera resolution, but encode a smaller preview
            # so JPEG work does not become the Nano bottleneck.
            preview = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA)
            encoded_ok, encoded = cv2.imencode(".jpg", preview, [cv2.IMWRITE_JPEG_QUALITY, 65])
            if not encoded_ok:
                continue
            pose = {
                "status": "person_candidate" if metrics is not None else "no_valid_pose",
                "state": posture_state,
                "message": message,
                "points": points,
                "fps": round(fps, 2),
                "latency_ms": round(latency_ms, 2),
                "updated_at": time.time(),
                "calibrated": baseline is not None,
                "calibration_phase": calibration_phase,
                "calibration_remaining": round(calibration_remaining, 1),
                "head_delta": None if head_delta is None else round(head_delta, 3),
                "torso_delta": None if torso_delta is None else round(torso_delta, 3),
                "confidence": None if steady_metrics is None else round(steady_metrics.confidence, 3),
                "side": None if steady_metrics is None else steady_metrics.side,
                "bad_for": round(bad_for, 1),
                "alert_count": state.alert_count,
                "events": list(state.events),
                "head_threshold": args.head_threshold,
                "torso_threshold": args.torso_threshold,
                "hold_seconds": args.hold_seconds,
            }
            state.publish(encoded.tobytes(), pose)
    except Exception as error:
        with state.condition:
            state.pose.update({
                "state": "OFFLINE",
                "message": "실행 오류: {}".format(error),
                "calibration_phase": "idle",
            })
            state.condition.notify_all()
    finally:
        if camera is not None and camera_open:
            camera.__exit__(None, None, None)
        if device is not None:
            device.close()


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class Handler(BaseHTTPRequestHandler):
    state = None
    index_html = ""

    def log_message(self, fmt, *args):
        return

    def send_bytes(self, data, content_type, status=200):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            self.send_bytes(self.index_html.encode("utf-8"), "text/html; charset=utf-8")
        elif path in ("/api/pose", "/api/status"):
            frame_id, _, pose = self.state.snapshot()
            pose["frame_id"] = frame_id
            self.send_bytes(json.dumps(pose).encode("utf-8"), "application/json; charset=utf-8")
        elif path in ("/api/health", "/healthz"):
            frame_id, jpeg, pose = self.state.snapshot()
            payload = {"ok": jpeg is not None, "frame_id": frame_id, "pose": pose}
            self.send_bytes(json.dumps(payload).encode("utf-8"), "application/json; charset=utf-8")
        elif path == "/video.mjpeg":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            last_id = -1
            try:
                while self.state.running:
                    with self.state.condition:
                        self.state.condition.wait_for(lambda: self.state.frame_id != last_id or not self.state.running, timeout=2.0)
                        frame_id, jpeg, _ = self.state.snapshot()
                    if jpeg is None or frame_id == last_id:
                        continue
                    last_id = frame_id
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(jpeg)).encode("ascii") + b"\r\n\r\n" + jpeg + b"\r\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            self.send_bytes(b"not found", "text/plain", 404)

    def do_POST(self):
        path = urlparse(self.path).path
        length = int(self.headers.get("Content-Length", "0"))
        if length:
            self.rfile.read(length)
        if path == "/api/calibrate":
            accepted, message = self.state.request_calibration()
            payload = json.dumps({"accepted": accepted, "message": message}, ensure_ascii=False)
            self.send_bytes(
                payload.encode("utf-8"),
                "application/json; charset=utf-8",
                202 if accepted else 409,
            )
        else:
            self.send_bytes(b"not found", "text/plain", 404)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", default="models/pose-resnet18-fp16.engine")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--baseline", default="data/posture_baseline.json")
    parser.add_argument("--events", default="logs/posture-events.jsonl")
    parser.add_argument("--warmup-seconds", type=float, default=3.0)
    parser.add_argument("--calibration-seconds", type=float, default=5.0)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--head-threshold", type=float, default=0.14)
    parser.add_argument("--torso-threshold", type=float, default=0.12)
    parser.add_argument("--hold-seconds", type=float, default=3.0)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    index_path = root.parent / "web" / "index.html"
    if not Path(args.engine).exists():
        raise SystemExit("TensorRT engine not found: {}".format(args.engine))
    state = State()
    Handler.state = state
    Handler.index_html = index_path.read_text()
    thread = threading.Thread(target=worker, args=(state, args), daemon=True)
    thread.start()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print("posture web MVP listening on http://{}:{}".format(args.host, args.port), flush=True)

    def stop_server(_signum, _frame):
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, stop_server)
    signal.signal(signal.SIGINT, stop_server)
    try:
        server.serve_forever()
    finally:
        state.running = False
        server.server_close()
        thread.join(timeout=3.0)


if __name__ == "__main__":
    main()
