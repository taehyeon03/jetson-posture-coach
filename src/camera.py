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
