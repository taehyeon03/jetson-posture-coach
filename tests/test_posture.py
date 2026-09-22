import unittest

from src.posture import Baseline, SustainedAlert, classify, extract_metrics


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


if __name__ == "__main__":
    unittest.main()
