from dataclasses import dataclass
import mediapipe as mp

def _clamp(v, lo, hi):
    return max(lo, min(hi, v))

def _bbox_from_normalized_landmarks(norm_landmarks, w: int, h: int, pad: int = 10):
    """
    norm_landmarks: finding some values of .x, .y in [0, 1]
    Return bbox in pixel coords: (x0, y0, x1, y1)
    """
    xs = [lm.x for lm in norm_landmarks]
    ys = [lm.y for lm in norm_landmarks]

    if not xs or not ys:
        return None

    x0 = int(min(xs) * w) - pad
    y0 = int(min(ys) * h) - pad
    x1 = int(max(xs) * w) + pad
    y1 = int(max(ys) * h) + pad

    x0 = _clamp(x0, 0, w - 1)
    y0 = _clamp(y0, 0, h - 1)
    x1 = _clamp(x1, 0, w - 1)
    y1 = _clamp(y1, 0, h - 1)

    return (x0, y0, x1, y1)


@dataclass
class HandDetectorConfig:
    model_path: str
    num_hands: int = 2
    min_hand_detection_confidence: float = 0.5
    min_hand_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    bbox_pad_px: int = 10



class HandDetector:
    """
    A wrapper for HandLandmarker

    """

    def __init__(self, cfg: HandDetectorConfig):
        self.cfg = cfg


        BaseOptions = mp.tasks.BaseOptions
        VisionRunningMode = mp.tasks.vision.RunningMode
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

        options = HandLandmarkerOptions(
            base_options = BaseOptions(model_asset_path = cfg.model_path),
            running_mode = VisionRunningMode.VIDEO,
            num_hands = cfg.num_hands,
            min_hand_detection_confidence = cfg.min_hand_detection_confidence,
            min_hand_presence_confidence = cfg.min_hand_presence_confidence,
            min_tracking_confidence = cfg.min_tracking_confidence,
            )

        self._detector = HandLandmarker.create_from_options(options)

    
    def close(self):
        # Mediapipe Tasks objects support close()
        self._detector.close()

    def detect(self, mp_image, timestamp_ms: int, image_w: int, image_h: int):
        """
        mediapipe.Image(SRGB) -> mp_image

        """

        result = self._detector.detect_for_video(mp_image, timestamp_ms)

         
        hand_boxes = []
        if result.hand_landmarks:
            for hand_lms in result.hand_landmarks:
                hb = _bbox_from_normalized_landmarks(
                        hand_lms,
                        w = image_w,
                        h = image_h,
                        pad = self.cfg.bbox_pad_px,
                        )
                if hb is not None:
                    hand_boxes.append(hb)

        return hand_boxes

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close() 
        return False



    

