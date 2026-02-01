from dataclasses import dataclass
import cv2
from ultralytics import YOLO

def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


@dataclass
class PhoneDetectorConfig:
    weights: str = "yolov8n.plt"
    conf: float = 0.5
    img_size: int = 640



class PhoneDetector:
    """
    Run YOLO and return phone boxes in original frame coordinates.
    Returns: list of (x0, y0, x1, y1, conf.)
    """
    
    def __init__(self, cfg: PhoneDetectorConfig):
        self.cfg = cfg
        self.model = YOLO(cfg.weights)


    def detect(self, frame_bgr):
        h, w = frame_bgr.shape[:2]

        # Resize for speed (keep aspect ratio by simple scaling)
        scale = self.cfg.img_size/max(h,w)
    
        if scale < 1.0:
            new_w = int(w * scale)
            new_h = int(h * scale)
            small = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            small = frame_bgr
            new_h, new_w = h, w
            scale = 1.0
    
        # Ultralytics expects BGR or RGB
        # verbose = False keeps the console clean.
        results = self.model.predict(small, conf = self.cfg.conf, verbose=False)
    
        phone_boxes = []
        r0 = results[0]
        if r0.boxes is None or len(r0.boxes) == 0:
            return phone_boxes
    
        names = r0.names # dict: class_id -> class_name
    
        # Iterate detections
        for box in r0.boxes:
            cls_id = int(box.cls.item())
            cls_name = names.get(cls_id, str(cls_id))
            if cls_name != "cell phone":
                continue
    
            conf = float(box.conf.item())
            x0, y0, x1, y1, = box.xyxy[0].tolist()
    
            # Map back to origianl frame coordinates
            if scale != 1.0:
                x0 = x0 / scale
                y0 = y0 / scale
                x1 = x1 / scale
                y1 = y1 / scale
    
            # Clamp + int
            x0 = _clamp(int(x0), 0, w - 1)
            y0 = _clamp(int(y0), 0, h - 1)
            x1 = _clamp(int(x1), 0, w - 1)
            y1 = _clamp(int(y1), 0, h - 1)
    
            phone_boxes.append((x0, y0, x1, y1, conf))
    
        return phone_boxes
