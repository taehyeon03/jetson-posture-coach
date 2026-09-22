import unittest

from src.posture import (
    Baseline,
    KeypointSmoother,
    OneEuroFilter,
    PostureMetrics,
    SteadyMetrics,
    SustainedAlert,
    classify,
    diagnose_pose,
    extract_metrics,
)


def pose(ear_x=110, shoulder_x=100, hip_x=100, nose_x=125):
    hidden = (0, 0, 0.0)
    return {
        "nose": (nose_x, 50, 0.9),
        "left_ear": (ear_x, 60, 0.9),
        "left_shoulder": (shoulder_x, 100, 0.9),
        "left_hip": (hip_x, 200, 0.9),
        "right_ear": hidden,
        "right_shoulder": hidden,
        "right_hip": hidden,
    }


class PostureTests(unittest.TestCase):
    def setUp(self):
        self.baseline = Baseline(0.10, 0.0, 20)

    def test_extracts_scale_independent_side_metrics(self):
        metrics = extract_metrics(pose())
        self.assertEqual(metrics.side, "left")
        self.assertAlmostEqual(metrics.head_forward, 0.10)
        self.assertAlmostEqual(metrics.torso_forward, 0.0)

    def test_classifies_head_and_torso_independently(self):
        state, _, _ = classify(extract_metrics(pose(ear_x=130, nose_x=145)), self.baseline)
        self.assertEqual(state, "HEAD_FORWARD")
        state, _, _ = classify(
            extract_metrics(pose(ear_x=150, shoulder_x=120, nose_x=165)), self.baseline
        )
        self.assertEqual(state, "HEAD_AND_TORSO_FORWARD")

    def test_rejects_missing_hip(self):
        points = pose()
        points["left_hip"] = (100, 200, 0.1)
        self.assertIsNone(extract_metrics(points))

    def test_alert_fires_once_after_hold_time(self):
        alert = SustainedAlert(3.0)
        self.assertEqual(alert.update("HEAD_FORWARD", 10.0)[0], False)
        self.assertEqual(alert.update("HEAD_FORWARD", 13.1)[0], True)
        self.assertEqual(alert.update("HEAD_FORWARD", 14.0)[0], False)
        alert.update("GOOD", 15.0)
        self.assertEqual(alert.update("HEAD_FORWARD", 20.0)[0], False)

    def test_diagnose_pose_names_the_missing_landmark(self):
        points = pose()
        points["left_hip"] = (100, 200, 0.1)
        diagnosis = diagnose_pose(points)
        self.assertEqual(diagnosis["side"], "left")
        self.assertEqual(diagnosis["missing"], ["hip"])

    def test_diagnose_pose_reports_nothing_missing_when_fully_visible(self):
        self.assertEqual(diagnose_pose(pose())["missing"], [])


class OneEuroFilterTests(unittest.TestCase):
    def test_first_sample_passes_through_unchanged(self):
        filt = OneEuroFilter()
        self.assertEqual(filt(5.0, 0.0), 5.0)

    def test_smooths_noise_around_a_constant_signal(self):
        filt = OneEuroFilter(min_cutoff=1.0, beta=0.0)
        noisy = [10.0, 10.4, 9.6, 10.3, 9.7, 10.2, 9.8, 10.1, 9.9, 10.0]
        smoothed = [filt(x, t * (1.0 / 30.0)) for t, x in enumerate(noisy)]
        raw_spread = max(noisy[3:]) - min(noisy[3:])
        smoothed_spread = max(smoothed[3:]) - min(smoothed[3:])
        self.assertLess(smoothed_spread, raw_spread)

    def test_tracks_a_fast_ramp_without_large_lag(self):
        filt = OneEuroFilter(min_cutoff=1.0, beta=1.0)
        value = 0.0
        for t in range(1, 61):
            value = float(t)
            output = filt(value, t * (1.0 / 30.0))
        self.assertGreater(output, 55.0)


class KeypointSmootherTests(unittest.TestCase):
    def test_smooths_position_but_passes_score_through(self):
        smoother = KeypointSmoother(min_cutoff=1.0, beta=0.0)
        first = smoother.smooth({"nose": (100.0, 50.0, 0.9)}, now=0.0)
        second = smoother.smooth({"nose": (108.0, 50.0, 0.4)}, now=1.0 / 30.0)
        self.assertEqual(first["nose"], (100.0, 50.0, 0.9))
        self.assertLess(second["nose"][0], 108.0)
        self.assertEqual(second["nose"][2], 0.4)

    def test_missing_point_passes_through_as_none(self):
        smoother = KeypointSmoother()
        result = smoother.smooth({"left_hip": None}, now=0.0)
        self.assertIsNone(result["left_hip"])


class SteadyMetricsTests(unittest.TestCase):
    def test_holds_last_reading_within_grace_window(self):
        steady = SteadyMetrics(grace_seconds=0.5)
        metrics = PostureMetrics("left", 0.1, 0.0, 0.9)
        self.assertIs(steady.update(metrics, now=0.0), metrics)
        self.assertIs(steady.update(None, now=0.3), metrics)

    def test_drops_to_none_after_grace_window_expires(self):
        steady = SteadyMetrics(grace_seconds=0.5)
        metrics = PostureMetrics("left", 0.1, 0.0, 0.9)
        steady.update(metrics, now=0.0)
        self.assertIsNone(steady.update(None, now=0.6))

    def test_fresh_metrics_always_overrides_held_state(self):
        steady = SteadyMetrics(grace_seconds=0.5)
        first = PostureMetrics("left", 0.1, 0.0, 0.9)
        second = PostureMetrics("left", 0.2, 0.0, 0.9)
        steady.update(first, now=0.0)
        self.assertIs(steady.update(second, now=0.1), second)


if __name__ == "__main__":
    unittest.main()
