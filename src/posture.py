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


class OneEuroFilter(object):
    """Adaptive low-pass filter for a noisy real-time scalar stream.

    Standard technique for stabilizing camera-based pose landmarks (used by
    MediaPipe's own landmark smoothing, among others): heavier smoothing
    while the signal is nearly still, lighter smoothing during fast motion,
    so posture no longer twitches between good/bad on raw single-frame
    keypoint jitter.
    """

    def __init__(self, min_cutoff=1.0, beta=0.3, d_cutoff=1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_previous = None
        self.dx_previous = 0.0
        self.t_previous = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        if self.t_previous is None:
            self.x_previous, self.t_previous = x, t
            return x
        dt = max(1e-6, t - self.t_previous)
        a_d = self._alpha(self.d_cutoff, dt)
        dx = (x - self.x_previous) / dt
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_previous
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1.0 - a) * self.x_previous
        self.x_previous, self.dx_previous, self.t_previous = x_hat, dx_hat, t
        return x_hat


class KeypointSmoother(object):
    """Runs a OneEuroFilter per axis per named keypoint over successive
    ``model.infer()`` outputs. Confidence scores pass through unsmoothed;
    only the (x, y) position is stabilized."""

    def __init__(self, min_cutoff=1.0, beta=0.3, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._filters = {}

    def smooth(self, points, now=None):
        now = time.time() if now is None else float(now)
        smoothed = {}
        for name, point in points.items():
            if point is None:
                smoothed[name] = point
                continue
            x, y, score = point
            if name not in self._filters:
                self._filters[name] = (
                    OneEuroFilter(self.min_cutoff, self.beta, self.d_cutoff),
                    OneEuroFilter(self.min_cutoff, self.beta, self.d_cutoff),
                )
            filter_x, filter_y = self._filters[name]
            smoothed[name] = (filter_x(x, now), filter_y(y, now), score)
        return smoothed


class PersonLock(object):
    """Keeps every frame anchored to the SAME person by gating on eye
    position, so a bystander walking through the background cannot hijack
    the ROI crop.

    Side view usually only shows one eye clearly (the far eye is occluded
    by the nose bridge), so the anchor is the single most-confident eye's
    position relative to the nose -- not the interocular midpoint, which
    needs both eyes and only works face-on. Nose-to-eye distance stands in
    for head size as the movement-budget scale, since that is what stays
    visible in profile.
    """

    def __init__(self, max_jump_ratio=3.0, reacquire_after=10):
        self.max_jump_ratio = float(max_jump_ratio)
        self.reacquire_after = int(reacquire_after)
        self.anchor = None
        self.lost_streak = 0

    @staticmethod
    def _anchor(points, minimum_confidence):
        nose = points.get("nose")
        if not _valid(nose, minimum_confidence):
            return None
        eyes = [points.get("left_eye"), points.get("right_eye")]
        eyes = [p for p in eyes if _valid(p, minimum_confidence)]
        if not eyes:
            return None
        eye = max(eyes, key=lambda p: p[2])
        scale = math.hypot(eye[0] - nose[0], eye[1] - nose[1])
        if scale < 2.0:
            return None
        return (eye[0], eye[1], scale)

    def update(self, points, minimum_confidence=0.25):
        """Call once per frame. Returns True if this frame's face belongs to
        the locked person (or starts a new lock), False if it jumped to
        someone else and the caller should treat the frame as unusable.
        """
        anchor = self._anchor(points, minimum_confidence)
        if anchor is None:
            self._miss()
            return False

        if self.anchor is None:
            self.anchor = anchor
            self.lost_streak = 0
            return True

        prev_x, prev_y, prev_scale = self.anchor
        x, y, scale = anchor
        jump = math.hypot(x - prev_x, y - prev_y)
        if jump > self.max_jump_ratio * max(prev_scale, scale, 1.0):
            self._miss()
            return False

        self.anchor = anchor
        self.lost_streak = 0
        return True

    def _miss(self):
        self.lost_streak += 1
        if self.lost_streak >= self.reacquire_after:
            self.anchor = None


class SteadyMetrics(object):
    """Bridges brief single-frame confidence drops (e.g. the hip flickering
    behind a chair edge) so the classifier does not snap to NO_POSE on every
    frame a keypoint dips under threshold. Only ever holds over a real prior
    reading, and only for ``grace_seconds``, so a genuine "person left" still
    reaches NO_POSE quickly.
    """

    def __init__(self, grace_seconds=0.6):
        self.grace_seconds = float(grace_seconds)
        self._last = None
        self._last_time = None

    def update(self, metrics, now=None):
        now = time.time() if now is None else float(now)
        if metrics is not None:
            self._last = metrics
            self._last_time = now
            return metrics
        if self._last is not None and now - self._last_time <= self.grace_seconds:
            return self._last
        return None


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

