from dataclasses import dataclass
import time
import cv2
from ultralytics import YOLO



@dataclass
class PhoneDetectorConfig:
    weights: str = "yolov8n.plt"
    conf: float = 0.4
    img_size: int = 640

    # Tracking config
    enable_tracking: bool = True
    tracker_type: str = "CSRT"          # "CSRT" or "KCF"
    min_init_conf: float = 0.6          # YOLO conf needed to (re)initialize tracker
    max_track_age_s: float = 1.0        # drop tracker if not re-confirm by YOLO within sometime
    min_track_box_area: int = 800       # drop tracker if box becomes too tiny
    yolo_every_n: int = 3               # run YOLO every N frames (tracking fills the gap)


    # Score system
    score_init: float = 0.0
    score_inc: float = 0.35             # If YOLO detects phone, score increase by a factor
    score_decay_tracker: float = 0.92   # decay when only tracker updates
    score_decay_miss: float = 0.80      # decay when neither Yolo nor tracker updates
    score_drop: float = 0.15            # if score below this, == no phone


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))

def _xyxy_to_xywh(x0, y0, x1, y1):
    return (x0, y0, max(1, x1 - x0), max(1, y1 - y0))

def _xywh_to_xyxy(x, y, w, h):
    return (x, y, x + w, y + h)

def _create_tracker(tracker_type: str):
    t = tracker_type.upper()
    if t == "CSRT":
        if hasattr(cv2, "TrackerCSRT_create"):
            print ("CSRT created")
            return cv2.TrackerCSRT_create()
        raise RuntimeError(
            "CSRT tracker not available in your OpenCV build. "
            "Install opencv-contrib-python."
        )
    if t == "KCF":
        if hasattr(cv2, "TrackerKCF_create"):
            return cv2.TrackerKCF_create()
        raise RuntimeError(
            "KCF tracker not available in your OpenCV build. "
            "Install opencv-contrib-python."
        )
    raise ValueError(f"Unsupported tracker_type={tracker_type!r}. Use 'CSRT' or 'KCF'.")


class PhoneDetector:
    """
    Run YOLO and return phone boxes in original frame coordinates.
    A tracker added
    Returns: list of (x0, y0, x1, y1, conf.)
    """
    
    def __init__(self, cfg: PhoneDetectorConfig):
        self.cfg = cfg
        self.model = YOLO(cfg.weights)

        self._tracker = None
        self._tracked_xyxy = None
        self._last_yolo_confirm_t = -1e9
        self._frame_i = 0

        self._score = cfg.score_init
        self._had_update_this_frame = False

    def _detect_yolo(self, frame_bgr):
        """Return phone boxes from YOLO only."""
        h, w = frame_bgr.shape[:2]

        # Resize for speed
        scale = self.cfg.img_size / max(h, w)
        if scale < 1.0:
            new_w = int(w * scale)
            new_h = int(h * scale)
            small = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            small = frame_bgr
            scale = 1.0

        results = self.model.predict(small, conf=self.cfg.conf, verbose=False)
        r0 = results[0]
        if r0.boxes is None or len(r0.boxes) == 0:
            return []

        names = r0.names
        boxes = []

        for box in r0.boxes:
            cls_id = int(box.cls.item())
            if names.get(cls_id, str(cls_id)) != "cell phone":
                continue

            conf = float(box.conf.item())
            x0, y0, x1, y1 = box.xyxy[0].tolist()

            if scale != 1.0:
                x0, y0, x1, y1 = x0 / scale, y0 / scale, x1 / scale, y1 / scale

            x0 = _clamp(int(x0), 0, w - 1)
            y0 = _clamp(int(y0), 0, h - 1)
            x1 = _clamp(int(x1), 0, w - 1)
            y1 = _clamp(int(y1), 0, h - 1)

            boxes.append((x0, y0, x1, y1, conf))

        # Sort highest confidence first
        boxes.sort(key=lambda b: b[4], reverse=True)
        return boxes

    def _maybe_init_or_update_tracker(self, frame_bgr, yolo_boxes, now_t: float):
        """
        Tracker policy:
        - If YOLO gives a high-confidence phone, (re)init tracker on best box.
        - Else, if tracker exists, update it.
        - Drop tracker if too old since last YOLO confirm or box is invalid.
        """
        self._had_update_this_frame = False

        h, w = frame_bgr.shape[:2]

        # (Re)initialize tracker if we have a strong YOLO detection
        if self.cfg.enable_tracking and yolo_boxes:
            best = yolo_boxes[0]
            x0, y0, x1, y1, conf = best
            if conf >= self.cfg.min_init_conf:
                self._tracker = _create_tracker(self.cfg.tracker_type)
                ok = self._tracker.init(frame_bgr, _xyxy_to_xywh(x0, y0, x1, y1))
                if ok:
                    self._tracked_xyxy = (x0, y0, x1, y1)
                    self._last_yolo_confirm_t = now_t
                    self._had_update_this_frame = True
                    print ("YOLO updated box")

        # Update tracker if YOLO didn't confirm this frame
        if self.cfg.enable_tracking and self._tracker is not None:
            ok, xywh = self._tracker.update(frame_bgr)
            if ok:
                x, y, ww, hh = [int(v) for v in xywh]
                x0, y0, x1, y1 = _xywh_to_xyxy(x, y, ww, hh)
                x0 = _clamp(x0, 0, w - 1)
                y0 = _clamp(y0, 0, h - 1)
                x1 = _clamp(x1, 0, w - 1)
                y1 = _clamp(y1, 0, h - 1)
                self._tracked_xyxy = (x0, y0, x1, y1)
                self._had_update_this_frame = True
            else:
                # Tracker failed this frame
                self._tracker = None
                self._tracked_xyxy = None

        # Drop stale tracker if not recently confirmed by YOLO
        if self._tracker is not None:
            age = now_t - self._last_yolo_confirm_t
            if age > self.cfg.max_track_age_s:
                self._tracker = None
                self._tracked_xyxy = None

        # Drop tiny/degenerate boxes
        if self._tracked_xyxy is not None:
            x0, y0, x1, y1 = self._tracked_xyxy
            area = max(0, x1 - x0) * max(0, y1 - y0)
            print (area)
            if area < self.cfg.min_track_box_area:
                self._tracker = None
                self._tracked_xyxy = None


    def detect(self, frame_bgr):
        """
        Returns phone boxes. Priority:
        - YOLO detection
        - Else tracked box if tracker is alive
        """

        self._frame_i += 1
        now_t = time.monotonic()

        run_yolo = True
        if self.cfg.enable_tracking and self.cfg.yolo_every_n > 1:
            # YOLO every N frames, tracker fills the gap
            run_yolo = (self._frame_i % self.cfg.yolo_every_n) == 0 or self._tracker is None

        yolo_boxes = self._detect_yolo(frame_bgr) if run_yolo else []

        # Update/init tracker using YOLO and/or tracking step
        self._maybe_init_or_update_tracker(frame_bgr, yolo_boxes, now_t)

        # Score based
        if yolo_boxes:
            # use best YOLO phone detection as evidence
            best_conf = float(yolo_boxes[0][4])
            # conf is already in [0, 1]; push score upward
            self._score = min(3.0, self._score + self.cfg.score_inc * best_conf)

        elif self._had_update_this_frame:
            # tracker updated bbox but no YOLO detection
            self._score *= self.cfg.score_decay_tracker
        else:
            # Neither Yolo nor tracker gave anything
            self._score *= self.cfg.score_decay_miss

        self._score = float(_clamp(self._score, 0.0, 3.0))

        # if score too low, drop everything
        if self._score < self.cfg.score_drop:
            self._tracker = None
            self._tracked_xyxy = None
            return []

        # Return
        if yolo_boxes:
            # Return best YOLO box but replace conf with score
            x0, y0, x1, y1, _ = yolo_boxes[0]
            return [(x0, y0, x1, y1, self._score)]

        if self._tracked_xyxy is not None:
            x0, y0, x1, y1 = self._tracked_xyxy
            # Syn here
            return [(x0, y0, x1, y1, self._score)]

        return []

