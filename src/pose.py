"""MoveNet pose inference with coordinates mapped back to the camera frame."""

from __future__ import print_function

import cv2
import numpy as np


KEYPOINT_NAMES = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)

SKELETON = (
    ("left_ear", "left_shoulder"),
    ("right_ear", "right_shoulder"),
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("right_hip", "right_knee"),
)


class MoveNet(object):
    """Small TensorFlow Lite wrapper for MoveNet SinglePose Lightning."""

    def __init__(self, model_path, threads=4):
        from tflite_runtime.interpreter import Interpreter

        try:
            self.interpreter = Interpreter(model_path=model_path, num_threads=threads)
        except TypeError:
            # TensorFlow Lite 2.1 on Jetson Nano does not expose num_threads.
            self.interpreter = Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input = self.interpreter.get_input_details()[0]
        self.output = self.interpreter.get_output_details()[0]
        shape = self.input["shape"]
        if tuple(shape) != (1, 192, 192, 3):
            raise ValueError("expected MoveNet Lightning input [1,192,192,3], got %r" % (shape,))
        self.size = int(shape[1])

    def infer(self, bgr_frame):
        """Return ``{name: (x_px, y_px, confidence)}`` for one BGR frame."""
        frame_h, frame_w = bgr_frame.shape[:2]
        scale = min(float(self.size) / frame_w, float(self.size) / frame_h)
        resized_w = max(1, int(round(frame_w * scale)))
        resized_h = max(1, int(round(frame_h * scale)))
        resized = cv2.resize(bgr_frame, (resized_w, resized_h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        top = (self.size - resized_h) // 2
        left = (self.size - resized_w) // 2
        canvas = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        canvas[top : top + resized_h, left : left + resized_w] = rgb
        tensor = np.expand_dims(canvas, axis=0).astype(self.input["dtype"])

        self.interpreter.set_tensor(self.input["index"], tensor)
        self.interpreter.invoke()
        raw = self.interpreter.get_tensor(self.output["index"])[0, 0]

        points = {}
        for name, value in zip(KEYPOINT_NAMES, raw):
            y_model, x_model, score = [float(v) for v in value]
            x = (x_model * self.size - left) / scale
            y = (y_model * self.size - top) / scale
            points[name] = (x, y, score)
        return points


def draw_pose(frame, points, confidence=0.25):
    """Draw detected joints on ``frame`` in place."""
    for first, second in SKELETON:
        a = points[first]
        b = points[second]
        if a[2] >= confidence and b[2] >= confidence:
            cv2.line(frame, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), (80, 220, 80), 2)
    for point in points.values():
        if point[2] >= confidence:
            cv2.circle(frame, (int(point[0]), int(point[1])), 4, (0, 220, 255), -1)

