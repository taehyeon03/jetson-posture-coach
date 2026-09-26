"""CSI camera capture for Jetson Nano via nvarguscamerasrc.

The onboard python3-opencv package on this board (apt, OpenCV 3.2) is built
without GStreamer support, and the CSI sensor exposes raw Bayer (RG10), so
cv2.VideoCapture(0) cannot read it directly. This module pulls frames through
GStreamer's appsink using PyGObject instead, bypassing cv2's video backend
entirely. Frames come out as plain BGR numpy arrays.
"""
import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst

import numpy as np


class CsiCamera:
    def __init__(self, sensor_id=0, width=1280, height=720, fps=30):
        Gst.init(None)
        self.width = width
        self.height = height
        pipeline_str = (
            f"nvarguscamerasrc sensor-id={sensor_id} ! "
            f"video/x-raw(memory:NVMM),width={width},height={height},framerate={fps}/1 ! "
            f"nvvidconv ! video/x-raw,format=BGRx ! "
            f"videoconvert ! video/x-raw,format=BGR ! "
            f"appsink name=sink emit-signals=false max-buffers=1 drop=true sync=false"
        )
        self._pipeline = Gst.parse_launch(pipeline_str)
        self._sink = self._pipeline.get_by_name("sink")
        # Best-effort capture timing of the last frame, for benchmarking:
        # buffer PTS and how old it was (pipeline running time - PTS) when read.
        self.last_pts_ns = None
        self.last_age_ns = None

    def __enter__(self):
        self._pipeline.set_state(Gst.State.PLAYING)
        return self

    def __exit__(self, *exc):
        self._pipeline.set_state(Gst.State.NULL)

    def read(self, timeout_ns=Gst.SECOND):
        """Return the latest BGR frame as an (H, W, 3) uint8 array, or None on timeout."""
        sample = self._sink.emit("try-pull-sample", timeout_ns)
        if sample is None:
            return None
        buf = sample.get_buffer()
        self._note_timing(buf)
        caps = sample.get_caps().get_structure(0)
        h = caps.get_value("height")
        w = caps.get_value("width")
        ok, mapinfo = buf.map(Gst.MapFlags.READ)
        if not ok:
            return None
        try:
            frame = np.frombuffer(mapinfo.data, dtype=np.uint8).reshape((h, w, 3)).copy()
        finally:
            buf.unmap(mapinfo)
        return frame

    def _note_timing(self, buf):
        self.last_pts_ns = None
        self.last_age_ns = None
        if buf.pts == Gst.CLOCK_TIME_NONE:
            return
        self.last_pts_ns = buf.pts
        clock = self._pipeline.get_clock()
        if clock is None:
            return
        age = clock.get_time() - self._pipeline.get_base_time() - buf.pts
        # A PTS that is not on the pipeline clock gives nonsense; drop it.
        if 0 <= age < 5 * Gst.SECOND:
            self.last_age_ns = age
