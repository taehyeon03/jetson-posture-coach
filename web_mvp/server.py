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
# The heatmap model has no notion of "a person" -- each joint channel is an
# independent argmax over the WHOLE frame, so with several people in view it
# can (and does) stitch one person's face onto another person's body. A face
# detector runs first to find every individual face, one is picked to track,
# and only the crop around THAT face is ever handed to the pose model, so it
# physically cannot see anyone else's joints.
FACE_CASCADE_PATH = Path(__file__).resolve().parent / "assets" / "haarcascade_profileface.xml"
FACE_DOWNSCALE = 0.4
FACE_MIN_SIZE = 24
FACE_LOST_RESET_FRAMES = 10
BODY_SIDE_MARGIN_RATIO = 3.0
BODY_TOP_MARGIN_RATIO = 1.5
BODY_BOTTOM_MARGIN_RATIO = 11.0

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


def infer(frame, context, device, arrays, pointers, input_i, outputs, heatmap_i, timing=None):
    """Run the engine on ``frame``. If ``timing`` is a dict it receives the
    per-stage milliseconds (pre_ms, h2d_ms, exec_ms, d2h_ms, decode_ms)."""
    t0 = time.perf_counter()
    _, _, height, width = arrays[input_i].shape
    rgb = cv2.cvtColor(cv2.resize(frame, (width, height)), cv2.COLOR_BGR2RGB)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    tensor = ((rgb.astype(np.float32) / 255.0 - mean) / std).transpose(2, 0, 1)[None]
    arrays[input_i][...] = tensor
    t1 = time.perf_counter()
    device.call("cuMemcpyHtoD_v2", pointers[input_i], ct.c_void_p(arrays[input_i].ctypes.data), ct.c_size_t(arrays[input_i].nbytes))
    t2 = time.perf_counter()
    if not context.execute_v2([p.value for p in pointers]):
        raise RuntimeError("TensorRT inference failed")
    t3 = time.perf_counter()
    for i in outputs:
        device.call("cuMemcpyDtoH_v2", ct.c_void_p(arrays[i].ctypes.data), pointers[i], ct.c_size_t(arrays[i].nbytes))
    t4 = time.perf_counter()
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
    if timing is not None:
        t5 = time.perf_counter()
        timing.update(
            pre_ms=(t1 - t0) * 1000.0,
            h2d_ms=(t2 - t1) * 1000.0,
            exec_ms=(t3 - t2) * 1000.0,
            d2h_ms=(t4 - t3) * 1000.0,
            decode_ms=(t5 - t4) * 1000.0,
        )
    return points


def load_face_cascade():
    cascade = cv2.CascadeClassifier(str(FACE_CASCADE_PATH))
    if cascade.empty():
        raise RuntimeError("cannot load face cascade: {}".format(FACE_CASCADE_PATH))
    return cascade


def detect_faces(frame, cascade, downscale=FACE_DOWNSCALE, min_size=FACE_MIN_SIZE):
    """Every distinct face in ``frame``, as full-frame pixel boxes.

    The profile cascade only recognizes one facing direction, so it also
    runs on a horizontally flipped copy to catch someone turned the other
    way (this side-view app never expects a frontal face).
    """
    small = cv2.resize(frame, None, fx=downscale, fy=downscale, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    size = (min_size, min_size)
    boxes = [
        (x / downscale, y / downscale, w / downscale, h / downscale)
        for (x, y, w, h) in cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=4, minSize=size)
    ]
    flipped = cv2.flip(gray, 1)
    small_w = gray.shape[1]
    for (x, y, w, h) in cascade.detectMultiScale(flipped, scaleFactor=1.2, minNeighbors=4, minSize=size):
        boxes.append(((small_w - x - w) / downscale, y / downscale, w / downscale, h / downscale))
    return boxes


def choose_face(boxes, anchor_center):
    """Pick the ONE face to track this frame: whichever is closest to where
    the locked person's face was last seen, or the largest (nearest-camera)
    face if nobody is locked yet."""
    if not boxes:
        return None
    if anchor_center is None:
        return max(boxes, key=lambda box: box[2] * box[3])
    ax, ay = anchor_center

    def distance(box):
        cx, cy = box[0] + box[2] / 2.0, box[1] + box[3] / 2.0
        return (cx - ax) ** 2 + (cy - ay) ** 2

    return min(boxes, key=distance)


def body_roi_from_face(face_box, frame_w, frame_h):
    """Expand one face box into a crop covering that sitting person's
    upper body -- generous enough for natural head/torso movement, sized
    off the face itself so it never reaches a neighboring desk."""
    x, y, w, h = face_box
    cx = x + w / 2.0
    x0 = max(0, int(cx - BODY_SIDE_MARGIN_RATIO * w))
    x1 = min(frame_w, int(cx + BODY_SIDE_MARGIN_RATIO * w))
    y0 = max(0, int(y - BODY_TOP_MARGIN_RATIO * h))
    y1 = min(frame_h, int(y + BODY_BOTTOM_MARGIN_RATIO * h))
    if x1 - x0 < 20 or y1 - y0 < 20:
        return None
    return (x0, y0, x1, y1)


def infer_tracked(frame, roi, context, device, arrays, pointers, input_i, outputs, heatmap_i, timing=None):
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

    raw_points = infer(crop, context, device, arrays, pointers, input_i, outputs, heatmap_i, timing)
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


def worker(state, args, recorder=None):
    """Capture/infer/publish loop. ``recorder`` (``src.bench.FrameRecorder``)
    optionally receives per-frame timings and detections for benchmarking."""
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
        face_cascade = load_face_cascade()
        calibration_phase = "idle"
        calibration_started = 0.0
        calibration_samples = []
        notice_message = None
        notice_until = 0.0
        previous = time.monotonic()
        measured = []
        locked_bbox = None
        face_anchor_center = None
        face_lost_streak = 0

        camera.__enter__()
        camera_open = True
        while state.running:
            read_began = time.perf_counter()
            frame = camera.read()
            if frame is None:
                if recorder is not None:
                    recorder.camera_timeout()
                continue
            t_start = time.perf_counter()
            started = time.monotonic()
            started_wall = time.time()
            frame_h, frame_w = frame.shape[:2]
            age_ns = camera.last_age_ns

            # Find every face, then keep tracking whichever one is closest
            # to the last locked position (or the biggest/nearest face if
            # nobody is locked yet), and crop to just that person's body.
            faces = detect_faces(frame, face_cascade)
            t_face = time.perf_counter()
            target_face = choose_face(faces, face_anchor_center)
            if target_face is not None:
                face_anchor_center = (target_face[0] + target_face[2] / 2.0, target_face[1] + target_face[3] / 2.0)
                face_lost_streak = 0
                locked_bbox = body_roi_from_face(target_face, frame_w, frame_h)
            else:
                face_lost_streak += 1
                if face_lost_streak >= FACE_LOST_RESET_FRAMES:
                    face_anchor_center = None
                    locked_bbox = None

            timing = {}
            points = infer_tracked(frame, locked_bbox, context, device, arrays, pointers, input_i, outputs, heatmap_i, timing)
            t_infer = time.perf_counter()
            point_map_raw = {
                p["name"]: (p["x"] * frame_w, p["y"] * frame_h, p["score"])
                for p in points
            }
            # Belt-and-suspenders sanity check even inside the face-locked
            # crop: a joint set that still doesn't line up with the locked
            # face is dropped rather than fed to the classifier.
            lock_ok = person_lock.update(point_map_raw, args.confidence)
            if lock_ok:
                point_map = smoother.smooth(point_map_raw, started)
                metrics = extract_metrics(point_map, args.confidence)
                # Send the smoothed positions to the browser too, so the
                # joint overlay itself stops twitching frame to frame.
                points = [
                    {"name": name, "x": x / frame_w, "y": y / frame_h, "score": score}
                    for name, (x, y, score) in point_map.items()
                ]
            else:
                point_map = point_map_raw
                metrics = None

            # Bridges a keypoint (usually the hip) dropping below threshold
            # for a frame or two, instead of the display flickering to
            # NO_POSE on every transient occlusion.
            steady_metrics = steady.update(metrics, started)
            t_track = time.perf_counter()

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

            t_state = time.perf_counter()
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
                "captured_at": started_wall - (0.0 if age_ns is None else age_ns / 1e9),
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
            if recorder is not None:
                t_end = time.perf_counter()
                total_ms = (t_end - t_start) * 1000.0
                record = {
                    "t": started,
                    "capture_wait_ms": (t_start - read_began) * 1000.0,
                    "age_ms": None if age_ns is None else age_ns / 1e6,
                    "face_ms": (t_face - t_start) * 1000.0,
                    "infer_ms": (t_infer - t_face) * 1000.0,
                    "track_ms": (t_track - t_infer) * 1000.0,
                    "state_ms": (t_state - t_track) * 1000.0,
                    "encode_ms": (t_end - t_state) * 1000.0,
                    "total_ms": total_ms,
                    "e2e_ms": None if age_ns is None else age_ns / 1e6 + total_ms,
                    "pts_ns": camera.last_pts_ns,
                    "face_found": target_face is not None,
                    "pose_valid": metrics is not None,
                    "lock_ok": lock_ok,
                    "state": posture_state,
                    "head_forward": None if steady_metrics is None else steady_metrics.head_forward,
                    "torso_forward": None if steady_metrics is None else steady_metrics.torso_forward,
                }
                record.update(timing)
                recorder.frame(record, point_map_raw, point_map)
                if recorder.frames and len(recorder.frames) % 30 == 1:
                    recorder.probe_gpu_memory()
    except Exception as error:
        if recorder is not None:
            recorder.error(error)
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
            pose["server_time"] = time.time()
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
