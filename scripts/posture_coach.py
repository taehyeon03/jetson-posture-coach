#!/usr/bin/env python3
"""Run personalized side-view posture detection on the Jetson CSI camera."""

from __future__ import print_function

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2

from src.camera import CsiCamera
from src.pose import MoveNet, draw_pose
from src.posture import Baseline, SustainedAlert, classify, extract_metrics


COLORS = {
    "GOOD": (60, 210, 60),
    "NO_POSE": (0, 180, 255),
    "HEAD_FORWARD": (0, 80, 255),
    "TORSO_FORWARD": (0, 80, 255),
    "HEAD_AND_TORSO_FORWARD": (0, 0, 255),
    "CALIBRATING": (255, 190, 0),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Personalized MoveNet posture coach")
    parser.add_argument("--model", default=str(ROOT / "models" / "movenet_singlepose_lightning.tflite"))
    parser.add_argument("--baseline", default=str(ROOT / "data" / "posture_baseline.json"))
    parser.add_argument("--calibrate", action="store_true", help="replace an existing baseline")
    parser.add_argument("--warmup-seconds", type=float, default=3.0)
    parser.add_argument("--calibration-seconds", type=float, default=5.0)
    parser.add_argument("--duration", type=float, default=30.0, help="monitoring seconds; 0 runs until Ctrl-C")
    parser.add_argument("--process-fps", type=float, default=8.0)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--head-threshold", type=float, default=0.14)
    parser.add_argument("--torso-threshold", type=float, default=0.12)
    parser.add_argument("--hold-seconds", type=float, default=3.0)
    parser.add_argument("--output", default=str(ROOT / "captures" / "posture-latest.jpg"))
    parser.add_argument("--events", default=str(ROOT / "logs" / "posture-events.jsonl"))
    parser.add_argument("--display", action="store_true", help="show a local preview; press q to stop")
    return parser.parse_args()


def overlay(frame, state, metrics, head_delta=None, torso_delta=None, detail=None):
    color = COLORS.get(state, (255, 255, 255))
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 92), (25, 25, 25), -1)
    cv2.putText(frame, "POSTURE: " + state, (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    if detail:
        cv2.putText(frame, detail, (18, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
    elif metrics is not None and head_delta is not None:
        line = "side=%s  head=%+.2f  torso=%+.2f  conf=%.2f" % (
            metrics.side,
            head_delta,
            torso_delta,
            metrics.confidence,
        )
        cv2.putText(frame, line, (18, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2)


def append_event(path, state, head_delta, torso_delta):
    parent = os.path.dirname(path)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)
    event = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "state": state,
        "head_delta": head_delta,
        "torso_delta": torso_delta,
    }
    with open(path, "a") as stream:
        stream.write(json.dumps(event, sort_keys=True) + "\n")


def main():
    args = parse_args()
    if not os.path.isfile(args.model):
        print("ERROR: model not found: %s" % args.model, file=sys.stderr)
        print("Run: bash scripts/install_posture_ai.sh", file=sys.stderr)
        return 2
    if args.process_fps <= 0:
        print("ERROR: --process-fps must be positive", file=sys.stderr)
        return 2

    model = MoveNet(args.model)
    needs_calibration = args.calibrate or not os.path.isfile(args.baseline)
    baseline = None if needs_calibration else Baseline.load(args.baseline)
    alert = SustainedAlert(args.hold_seconds)
    interval = 1.0 / args.process_fps
    last_inference = 0.0
    last_report = 0.0
    last_frame = None
    last_state = "NO_POSE"

    if needs_calibration:
        print("CALIBRATION: sit upright in side view; head, shoulder, and hip must be visible.")
        print("Calibration starts in %.1f seconds and lasts %.1f seconds." % (args.warmup_seconds, args.calibration_seconds))
    else:
        print("Loaded baseline: %s (%d samples)" % (args.baseline, baseline.samples))

    calibration_samples = []
    opened_at = time.time()
    calibration_start = opened_at + args.warmup_seconds
    monitoring_start = None
    processed = 0

    try:
        with CsiCamera(width=args.width, height=args.height, fps=args.camera_fps) as camera:
            while True:
                frame = camera.read()
                if frame is None:
                    continue
                now = time.time()
                if now - last_inference < interval:
                    continue
                last_inference = now
                processed += 1
                points = model.infer(frame)
                metrics = extract_metrics(points, args.confidence)
                draw_pose(frame, points, args.confidence)

                if baseline is None:
                    if now < calibration_start:
                        remaining = calibration_start - now
                        last_state = "CALIBRATING"
                        overlay(frame, last_state, metrics, detail="Sit upright - starts in %.1fs" % remaining)
                    elif now < calibration_start + args.calibration_seconds:
                        if metrics is not None:
                            calibration_samples.append(metrics)
                        remaining = calibration_start + args.calibration_seconds - now
                        last_state = "CALIBRATING"
                        overlay(frame, last_state, metrics, detail="Hold still - %.1fs (%d samples)" % (remaining, len(calibration_samples)))
                    else:
                        minimum_samples = max(8, int(args.calibration_seconds * args.process_fps * 0.35))
                        if len(calibration_samples) < minimum_samples:
                            print("ERROR: calibration found only %d valid samples (need %d)." % (len(calibration_samples), minimum_samples), file=sys.stderr)
                            print("Move the camera to the side and include head, shoulder, and hip.", file=sys.stderr)
                            overlay(
                                frame,
                                "NO_POSE",
                                metrics,
                                detail="Calibration failed - show head, shoulder, and hip",
                            )
                            last_frame = frame
                            break
                        baseline = Baseline.from_samples(calibration_samples)
                        baseline.save(args.baseline)
                        monitoring_start = now
                        print("Calibration saved: head=%.3f torso=%.3f samples=%d" % (baseline.head_forward, baseline.torso_forward, baseline.samples))
                        continue
                else:
                    if monitoring_start is None:
                        monitoring_start = now
                    state, head_delta, torso_delta = classify(
                        metrics,
                        baseline,
                        args.head_threshold,
                        args.torso_threshold,
                    )
                    last_state = state
                    fired, bad_seconds = alert.update(state, now)
                    overlay(frame, state, metrics, head_delta, torso_delta)
                    if fired:
                        print("\aALERT: %s continued for %.1fs" % (state, bad_seconds))
                        append_event(args.events, state, head_delta, torso_delta)
                    if now - last_report >= 1.0:
                        if state == "NO_POSE":
                            print("state=NO_POSE (show head, shoulder, and hip from the side)")
                        else:
                            print("state=%s head=%+.3f torso=%+.3f bad_for=%.1fs" % (state, head_delta, torso_delta, bad_seconds))
                        last_report = now

                last_frame = frame
                if args.display:
                    cv2.imshow("Jetson Posture Coach", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                if baseline is not None and args.duration > 0 and now - monitoring_start >= args.duration:
                    break
    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        if args.display:
            cv2.destroyAllWindows()

    if last_frame is not None and args.output:
        parent = os.path.dirname(args.output)
        if parent and not os.path.isdir(parent):
            os.makedirs(parent)
        cv2.imwrite(args.output, last_frame)
        print("Saved annotated frame: %s" % args.output)
    elapsed = max(0.001, time.time() - opened_at)
    print(
        "Finished: state=%s processed_frames=%d average_fps=%.2f"
        % (last_state, processed, processed / elapsed)
    )
    return 0 if baseline is not None else 2


if __name__ == "__main__":
    sys.exit(main())
