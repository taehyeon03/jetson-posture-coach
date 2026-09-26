import unittest

from src.bench import (
    FrameRecorder,
    SystemSampler,
    distribution,
    flatten_summary,
    linear_slope,
    parse_tegrastats,
    percentile,
    summarize,
)

NANO_LINE = (
    "RAM 1590/3956MB (lfb 224x4MB) SWAP 12/1978MB (cached 0MB) CPU [10%@1479,off,5%@1479,7%@1479] "
    "EMC_FREQ 0% GR3D_FREQ 37% PLL@34C CPU@37C PMIC@100C GPU@35.5C AO@43.5C thermal@36.5C "
    "POM_5V_IN 1897/1850 POM_5V_GPU 41/41 POM_5V_CPU 331/331"
)
ORIN_LINE = "RAM 2000/7620MB GR3D_FREQ 0%@[624] cpu@45.3C tj@46.1C VDD_IN 4162mW/4100mW VDD_CPU_GPU_CV 1119mW/1119mW"


def record(t, total=50.0, **extra):
    base = {
        "t": t,
        "total_ms": total,
        "e2e_ms": total + 10.0,
        "infer_ms": 30.0,
        "face_found": True,
        "pose_valid": True,
        "lock_ok": True,
        "state": "GOOD",
        "head_forward": 0.1,
        "torso_forward": 0.05,
    }
    base.update(extra)
    return base


def pose(shift=0.0, score=0.9):
    return {"nose": (100.0 + shift, 50.0, score), "left_ear": (90.0 + shift, 60.0, score)}


class StatsTests(unittest.TestCase):
    def test_percentile_interpolates(self):
        data = [1.0, 2.0, 3.0, 4.0]
        self.assertEqual(percentile(data, 0), 1.0)
        self.assertEqual(percentile(data, 100), 4.0)
        self.assertAlmostEqual(percentile(data, 50), 2.5)
        self.assertIsNone(percentile([], 50))

    def test_distribution_skips_none(self):
        result = distribution([10, None, 20, 30])
        self.assertEqual(result["count"], 3)
        self.assertEqual(result["p50"], 20.0)
        self.assertIsNone(distribution([None]))

    def test_linear_slope(self):
        self.assertAlmostEqual(linear_slope([0, 1, 2], [5, 7, 9]), 2.0)
        self.assertIsNone(linear_slope([1], [1]))


class TegrastatsTests(unittest.TestCase):
    def test_nano_line(self):
        parsed = parse_tegrastats(NANO_LINE)
        self.assertEqual(parsed["power_mw"], 1897.0)
        self.assertEqual(parsed["gpu_util_pct"], 37.0)
        self.assertNotIn("pmic", parsed["temps"])
        self.assertEqual(parsed["temp_max_c"], 43.5)

    def test_orin_line(self):
        parsed = parse_tegrastats(ORIN_LINE)
        self.assertEqual(parsed["power_mw"], 4162.0)
        self.assertEqual(parsed["temp_max_c"], 46.1)

    def test_unrelated_line(self):
        parsed = parse_tegrastats("garbage")
        self.assertIsNone(parsed["power_mw"])
        self.assertIsNone(parsed["temp_max_c"])


class RecorderTests(unittest.TestCase):
    def test_jitter_only_between_locked_frames(self):
        recorder = FrameRecorder(0.25)
        recorder.frame(record(0.0), pose(0.0), pose(0.0))
        recorder.frame(record(1.0), pose(4.0), pose(1.0))
        recorder.frame(record(2.0, lock_ok=False), pose(50.0), pose(50.0))
        recorder.frame(record(3.0), pose(0.0), pose(0.0))
        frames = recorder.frames
        self.assertIsNone(frames[0]["jitter_raw_px"])
        self.assertAlmostEqual(frames[1]["jitter_raw_px"], 4.0)
        self.assertAlmostEqual(frames[1]["jitter_smooth_px"], 1.0)
        self.assertIsNone(frames[2]["jitter_raw_px"])
        self.assertIsNone(frames[3]["jitter_raw_px"])

    def test_low_confidence_joints_excluded(self):
        recorder = FrameRecorder(0.25)
        recorder.frame(record(0.0), pose(0.0, 0.1), pose(0.0, 0.1))
        recorder.frame(record(1.0), pose(9.0, 0.1), pose(9.0, 0.1))
        self.assertIsNone(recorder.frames[1]["jitter_raw_px"])


def build_recorder(seconds=100, fps=10, latency=lambda i: 40.0 + (i % 5)):
    recorder = FrameRecorder(0.25)
    for i in range(seconds * fps):
        recorder.frame(
            record(
                i / float(fps),
                total=latency(i),
                pts_ns=int(i / float(fps) * 1e9),  # camera at 30 fps, processing at 10 fps
                face_found=(i % 10 != 0),
                pose_valid=(i % 20 != 0),
            ),
            pose(0.0),
            pose(0.0),
        )
    return recorder


def steady_samples(power=3000.0):
    return (
        [{"t_s": t, "phase": "idle", "rss_mb": 100.0, "ram_used_mb": 900.0, "swap_used_mb": 0.0, "temp_max_c": 40.0, "gpu_util_pct": 0.0, "power_mw": 1000.0} for t in range(-5, 0)]
        + [{"t_s": t, "phase": "steady", "rss_mb": 200.0 + t * 0.1, "ram_used_mb": 1500.0, "swap_used_mb": 0.0, "temp_max_c": 60.0, "gpu_util_pct": 50.0, "power_mw": power} for t in range(10, 100)]
    )


OPTIONS = {"confidence": 0.25, "deadline_ms": 200.0, "camera_fps": 30, "temp_limit_c": 80.0, "price_per_kwh": 200.0, "gpu_extra": {}}


class SummaryTests(unittest.TestCase):
    def test_summary_values(self):
        recorder = build_recorder()
        summary = summarize(recorder, steady_samples(), {"steady_start": 10.0, "steady_end": 100.0}, OPTIONS)
        self.assertAlmostEqual(summary["throughput"]["fps_mean"], 10.0, delta=0.2)
        self.assertEqual(summary["latency_ms"]["e2e_ms"]["min"], 50.0)
        self.assertEqual(summary["latency_ms"]["e2e_ms"]["max"], 54.0)
        self.assertAlmostEqual(summary["recognition"]["face_detect_rate"], 0.9, delta=0.02)
        self.assertAlmostEqual(summary["recognition"]["pose_valid_rate"], 0.95, delta=0.02)
        self.assertEqual(summary["reliability"]["deadline_miss_rate"], 0.0)
        self.assertAlmostEqual(summary["reliability"]["camera_frames_skipped_ratio"], 2.0 / 3.0, delta=0.02)
        self.assertEqual(summary["stability"]["state_flips"], 0)
        self.assertTrue(all(check["pass"] for check in summary["targets"]))

    def test_energy_per_inference(self):
        recorder = build_recorder()
        summary = summarize(recorder, steady_samples(3000.0), {"steady_start": 10.0, "steady_end": 100.0}, OPTIONS)
        energy = summary["energy"]
        # 3 W over 90 s = 270 J, spread over ~900 frames.
        self.assertAlmostEqual(energy["energy_j"], 270.0, delta=0.1)
        self.assertAlmostEqual(energy["energy_per_inference_mj"], 300.0, delta=5.0)
        self.assertAlmostEqual(energy["net_energy_per_inference_mj"], 200.0, delta=5.0)
        self.assertIn("cost_per_million_inferences", energy)

    def test_no_power_source(self):
        recorder = build_recorder(seconds=20)
        samples = [dict(s, power_mw=None) for s in steady_samples()]
        summary = summarize(recorder, samples, {"steady_start": 10.0, "steady_end": 20.0}, OPTIONS)
        self.assertFalse(summary["energy"]["available"])

    def test_slow_run_fails_targets_and_deadline(self):
        recorder = build_recorder(latency=lambda i: 400.0)
        summary = summarize(recorder, steady_samples(), {"steady_start": 10.0, "steady_end": 100.0}, OPTIONS)
        self.assertEqual(summary["reliability"]["deadline_miss_rate"], 1.0)
        self.assertFalse(summary["targets"][0]["pass"])

    def test_drift_detects_slowdown(self):
        recorder = build_recorder(latency=lambda i: 40.0 if i < 500 else 80.0)
        summary = summarize(recorder, steady_samples(), {"steady_start": 0.0, "steady_end": 100.0}, OPTIONS)
        self.assertGreater(summary["drift"]["p95_change_pct"], 50.0)

    def test_recorded_errors_marked_crashed(self):
        recorder = build_recorder(seconds=20)
        recorder.error(RuntimeError("boom"))
        summary = summarize(recorder, steady_samples(), {"steady_start": 10.0, "steady_end": 20.0}, OPTIONS)
        self.assertTrue(summary["reliability"]["crashed"])
        self.assertIn("boom", summary["reliability"]["errors"][0])

    def test_flatten(self):
        recorder = build_recorder(seconds=20)
        summary = summarize(recorder, steady_samples(), {"steady_start": 10.0, "steady_end": 20.0}, OPTIONS)
        row = flatten_summary("id", "label", summary)
        self.assertEqual(row["run_id"], "id")
        self.assertIsNotNone(row["latency_p95_ms"])

    def test_model_mode_has_no_recognition(self):
        recorder = build_recorder(seconds=20)
        options = dict(OPTIONS, mode="model")
        summary = summarize(recorder, steady_samples(), {"steady_start": 10.0, "steady_end": 20.0}, options)
        self.assertIsNone(summary["recognition"])
        self.assertEqual(flatten_summary("id", "", summary)["face_detect_rate"], None)


class SamplerTests(unittest.TestCase):
    def test_sampler_records_phases(self):
        sampler = SystemSampler(interval=0.05, use_tegrastats=False)
        sampler.start()
        sampler.mark("steady")
        import time

        time.sleep(0.2)
        sampler.stop()
        self.assertGreaterEqual(len(sampler.samples), 3)
        self.assertIn("steady", [s["phase"] for s in sampler.samples])


if __name__ == "__main__":
    unittest.main()
