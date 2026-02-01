from dataclasses import dataclass
import mediapipe as mp

@dataclass
class FaceDetectorConfig:
    model_path: str
    min_detection_confidence: float = 0.8

class FacePresenceDetector:
    """
    A wrapper for face detector
    """

    def __init__(self, cfg: FaceDetectorConfig):
        self.cfg = cfg

        BaseOptions = mp.tasks.BaseOptions
        FaceDetector = mp.tasks.vision.FaceDetector
        FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = FaceDetectorOptions(
                base_options = BaseOptions(model_asset_path = cfg.model_path),
                running_mode = VisionRunningMode.VIDEO,
                min_detection_confidence = cfg.min_detection_confidence,
                )

        self._detector = FaceDetector.create_from_options(options)

    
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def close(self):
        self._detector.close()

    def detect(self, mp_image, timestamp_ms: int):
        
        result = self._detector.detect_for_video(mp_image, timestamp_ms)
        detections = result.detections if result else None
        face_present = bool(detections)
        return face_present, detections
