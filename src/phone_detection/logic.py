from dataclasses import dataclass

# geometry calculation

def intersects(a, b) -> bool:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return not(ax1 < bx0 or bx1 < ax0 or ay1 < by0 or by1 < ay0)

def iou(a, b) -> float:

    """
    Intersection over Union for pixel bboxes a, b = (x0, y0, x1, y1)
    """

    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b

    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)

    iw = max(0, ix1 - ix0)
    ih = max(0, iy1 - iy0)
    inter = iw * ih

    area_a = max(0, ax1 - ax0) * max(0, ay1 - ay0)
    area_b = max(0, bx1 - bx0) * max(0, by1 - by0)
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0

# If hand-phone are near
def phone_held_by_hand(hand_boxes, phone_boxes, iou_thres=0.02) -> bool:
    """
    Return True if any phone bbox overlaps/intersects any hand bbox.

    Logic:
    - IOU > iou_thres (very small value tho) OR
    - intersection (if bboxes are thin)
    """

    for hb in hand_boxes:
        for (px0, py0, px1, py1, _conf) in phone_boxes:
            pb = (px0, py0, px1, py1)
            if intersects(hb, pb) and iou(hb, pb) >= iou_thres:

                return True
            if intersects(hb, pb) and iou_thres <= 0.0:
                #case of IOU small (some view angle of phone)
                return True

    return False

## State Machine
@dataclass
class SmoothingConfig:
    phone_on_frames: int = 5
    phone_off_frames: int = 10



class PhoneUseStateMachine:
    """
    For maintaining state based on face_present + phone_held.
    """

    def __init__(self, cfg: SmoothingConfig):
        self.cfg = cfg
        self.state = "AWAY"
        self._on = 0
        self._off = 0

    def reset(self):
        self.state = "AWAY"
        self._on = 0
        self._off = 0

    def update(self, face_present: bool, phone_held: bool) -> str:
        """
        Update counters and return state
        """

        if not face_present:
            self.state = "AWAY"
            self._on = 0
            self._off = 0
            return self.state
        
        # Face present
        if phone_held:
            self._on += 1
            self._off = 0
        else:
            self._off += 1
            self._on = 0

        if self._on >= self.cfg.phone_on_frames:
            self.state = "PHONE_USE"
        elif self._off >= self.cfg.phone_off_frames:
            self.state = "WORKING"

        return self.state




