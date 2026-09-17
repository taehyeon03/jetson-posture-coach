"""Board smoke test: confirm the CSI camera pipeline produces real frames at a usable rate.

Run directly on the Jetson Nano with system python3 (not a venv — this needs the
apt-installed PyGObject/GStreamer bindings, which are not pip-installable here):

    python3 scripts/camera_smoke_test.py [--seconds 5] [--save out.jpg]
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.camera import CsiCamera


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--save", type=str, default=None, help="save the last frame as a JPEG")
    args = parser.parse_args()

    frames = 0
    last_frame = None
    t0 = time.time()
    with CsiCamera(width=args.width, height=args.height, fps=args.fps) as cam:
        while time.time() - t0 < args.seconds:
            frame = cam.read()
            if frame is None:
                continue
            last_frame = frame
            frames += 1
    elapsed = time.time() - t0

    print(f"frames={frames} elapsed={elapsed:.2f}s fps={frames / elapsed:.2f} shape={None if last_frame is None else last_frame.shape}")

    if args.save and last_frame is not None:
        import cv2

        cv2.imwrite(args.save, last_frame)
        print(f"saved {args.save}")


if __name__ == "__main__":
    main()
