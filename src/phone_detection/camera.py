from dataclasses import dataclass
import cv2


@dataclass
class CameraConfig:
    index: int = 0
    flip_horizontal: bool = True
    resize_max: int | None = None

class Camera:
    """
    OpenCV camera open/preprocess wrapper
    """

    def __init__(self, cfg: CameraConfig):
        self.cfg = cfg
        self.cap = cv2.VideoCapture(cfg.index)
        if not self.cap.isOpened():
            raise RuntimeError(f"ERROR: Could not open camera index {cfg.index}.\n\n\n")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.release()
        return False

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def read(self):
        """
        Return a preprocessed BGR frame, or None if capture failed.
        """
        ret, frame = self.cap.read()
        if not ret:
            return None

        if self.cfg.flip_horizontal:
            frame = cv2.flip(frame, 1)

        if self.cfg.resize_max is not None:
            h, w  = frame.shape[:2]
            m = max(h, w)
            if m > self.cfg.resize_max:
                scale = self.cfg.resize_max / m
                new_w = int(w * scale)
                new_h = int(h * scale)

                frame = cv2.resize(frame, (new_w, new_h), interpolation = cv2.INTER_AREA)
        return frame
