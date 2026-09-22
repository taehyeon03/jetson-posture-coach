"""Personalized side-view posture metrics and sustained-state logic."""

from __future__ import print_function

import json
import math
import os
import statistics
import time


class PostureMetrics(object):
    def __init__(self, side, head_forward, torso_forward, confidence):
        self.side = side
        self.head_forward = float(head_forward)
        self.torso_forward = float(torso_forward)
        self.confidence = float(confidence)


def _valid(point, minimum):
    return point is not None and point[2] >= minimum


def extract_metrics(points, minimum_confidence=0.25):
    """Build scale-independent metrics from the most visible body side.

    Positive values point toward the direction the face is looking. Distances
    are divided by shoulder-to-hip length, so moving the chair slightly does
    not change the thresholds.
    """
    nose = points.get("nose")
    if not _valid(nose, minimum_confidence):
        return None

    candidates = []
    for side in ("left", "right"):
        ear = points.get(side + "_ear")
        shoulder = points.get(side + "_shoulder")
        hip = points.get(side + "_hip")
        if not all(_valid(p, minimum_confidence) for p in (ear, shoulder, hip)):
            continue
        confidence = min(nose[2], ear[2], shoulder[2], hip[2])
        candidates.append((confidence, side, ear, shoulder, hip))
    if not candidates:
        return None

    confidence, side, ear, shoulder, hip = max(candidates, key=lambda item: item[0])
    torso_x = shoulder[0] - hip[0]
    torso_y = shoulder[1] - hip[1]
    torso_length = math.sqrt(torso_x * torso_x + torso_y * torso_y)
    if torso_length < 20.0 or hip[1] <= shoulder[1]:
        return None

    facing = 1.0 if nose[0] >= ear[0] else -1.0
    head_forward = facing * (ear[0] - shoulder[0]) / torso_length
    torso_forward = facing * (shoulder[0] - hip[0]) / torso_length
    return PostureMetrics(side, head_forward, torso_forward, confidence)


def diagnose_pose(points, minimum_confidence=0.25):
    """Report which side-view landmarks are too low-confidence to use.

    Picks the side (left/right) with the fewest missing landmarks so the
    caller can tell the user exactly what to fix (e.g. only the hip is
    unreliable) instead of a generic "not visible" message.
    """
    nose = points.get("nose")
    missing = ["nose"] if not _valid(nose, minimum_confidence) else []

    best_side, best_missing = None, None
    for side in ("left", "right"):
        side_missing = [
            part
            for part in ("ear", "shoulder", "hip")
            if not _valid(points.get(side + "_" + part), minimum_confidence)
        ]
        if best_missing is None or len(side_missing) < len(best_missing):
            best_side, best_missing = side, side_missing

    missing.extend(best_missing)
    return {"side": best_side, "missing": missing}


class Baseline(object):
    VERSION = 1

    def __init__(self, head_forward, torso_forward, samples):
        self.head_forward = float(head_forward)
        self.torso_forward = float(torso_forward)
        self.samples = int(samples)

    @classmethod
    def from_samples(cls, metrics):
        if not metrics:
            raise ValueError("cannot calibrate without valid pose samples")
        return cls(
            statistics.median([m.head_forward for m in metrics]),
            statistics.median([m.torso_forward for m in metrics]),
            len(metrics),
        )

    def save(self, path):
        parent = os.path.dirname(path)
        if parent and not os.path.isdir(parent):
            os.makedirs(parent)
        payload = {
            "version": self.VERSION,
            "head_forward": self.head_forward,
            "torso_forward": self.torso_forward,
            "samples": self.samples,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        temporary = path + ".tmp"
        with open(temporary, "w") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.rename(temporary, path)

    @classmethod
    def load(cls, path):
        with open(path) as stream:
            payload = json.load(stream)
        if payload.get("version") != cls.VERSION:
            raise ValueError("unsupported posture baseline version")
        return cls(payload["head_forward"], payload["torso_forward"], payload["samples"])


def classify(metrics, baseline, head_threshold=0.14, torso_threshold=0.12):
    if metrics is None:
        return "NO_POSE", None, None
    head_delta = metrics.head_forward - baseline.head_forward
    torso_delta = metrics.torso_forward - baseline.torso_forward
    head_bad = head_delta >= head_threshold
    torso_bad = torso_delta >= torso_threshold
    if head_bad and torso_bad:
        state = "HEAD_AND_TORSO_FORWARD"
    elif head_bad:
        state = "HEAD_FORWARD"
    elif torso_bad:
        state = "TORSO_FORWARD"
    else:
        state = "GOOD"
    return state, head_delta, torso_delta


class SustainedAlert(object):
    """Emit one alert when the same bad state lasts for ``hold_seconds``."""

    def __init__(self, hold_seconds=3.0):
        self.hold_seconds = float(hold_seconds)
        self.state = None
        self.since = None
        self.alerted = False

    def update(self, state, now=None):
        now = time.time() if now is None else float(now)
        if state in ("GOOD", "NO_POSE"):
            self.state = state
            self.since = None
            self.alerted = False
            return False, 0.0
        if state != self.state:
            self.state = state
            self.since = now
            self.alerted = False
        elapsed = max(0.0, now - self.since)
        if elapsed >= self.hold_seconds and not self.alerted:
            self.alerted = True
            return True, elapsed
        return False, elapsed

