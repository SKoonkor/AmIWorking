# Python libraries

import time
from pathlib import Path

import cv2
import mediapipe as mp


def _resolve_model_path() -> str:
    """
    Resolve models/fact_detector.model relative to the repo root.
    """

    this_file = Path(__file__).resolve()
    repo_root = this_file.parents[2] # main.py -> phone_detection -> src -> ropo_root
    model_path = repo_root / "models" / "blaze_face_short_range.tflite"

    return str(model_path)


def _draw_detection(frame_bgr, detections):
    """
    Draw face bounding boxes + confidence on the frame
    """

    h, w  = frame_bgr.shape[:2]

    for det in detections:
        # det.bounding_box is basically the input image coordinations.
        bbox = det.bounding_box
        x0 = int(bbox.origin_x)
        y0 = int(bbox.origin_y)
        x1 = int(bbox.origin_x + bbox.width)
        y1 = int(bbox.origin_y + bbox.height)

        # Clamp to frame bounds (defensive)
        x0 = max(0, min(w - 1, x0))
        y0 = max(0, min(h - 1, y0))
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))

        cv2.rectangle(frame_bgr, (x0, y0), (x1, y1), (0, 255, 0), 2)


        # Confidence score is usually in categories[0].score

        score = None
        if det.categories:
            score = det.categories[0].score

        label = f"Face {score:.2f}" if score is not None else "Face"

        cv2.putText(
                frame_bgr,
                label,
                (x0, max(0, y0 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2, 
                cv2.LINE_AA,
                )

def main():
    """
    Main programme for phone detection project.

    Scheme:
    - Open face_detector model (Google AI.......);
    - Open the default webcam (maybe a webcam switch function later tho) 
    - Quite function (by pressing some key or a combination of keys)
    """


    model_path = _resolve_model_path()
    if not Path(model_path).exists():
        raise FileNotFoundError(
                f"Could not find model file at:\n {model_path}\n\n"
                "Creat <repo>/models and put face_detector inside it.")

    # Open default camera (usually index 0)
    cap = cv2.VideoCapture(0)
        # Change the index for different camera if there is any, later tho

    # Check if the camera opened successfully 
    if not cap.isOpened():
        raise RuntimeError("ERROR: Could not open camera.")

    print ("Camera opened successfully. Press 'q' to quite.")



    # MediaPipe setup
    BaseOptions = mp.tasks.BaseOptions
    FaceDetector = mp.tasks.vision.FaceDetector
    FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = FaceDetectorOptions(
            base_options = BaseOptions(model_asset_path=model_path),
            running_mode = VisionRunningMode.VIDEO,
            min_detection_confidence = 0.85,
            ) #Change min detection conf to more if want more strict detection

    start_ms = int(time.monotonic()*1000)
    
    with FaceDetector.create_from_options(options) as detector:

        print ("Running MediaPipe Task Face Detection. Prease q to quite")


        while True:
            # Read a frame from the webcam
            ret, frame_bgr = cap.read()

            if not ret:
                print ("WARNING: Failed to grab frame.")
                break

            # MediaPipe expects RGB
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


            # Wrap in mp.Image (SRGB)
            mp_image = mp.Image(image_format = mp.ImageFormat.SRGB, data = frame_rgb)

            # Timestamp required for VIDEO mode (in ms)
            timestamp_ms = int(time.monotonic() * 1000) - start_ms
                
            # Run detection
            result = detector.detect_for_video(mp_image, timestamp_ms)



            # Draw Detections
            if result.detections:
                _draw_detection(frame_bgr, result.detections)
                status = f"Faces: {len(result.detections)}"
                color = (0, 255, 0)

            else:
                status = "No face"
                color = (0, 0, 255)

            cv2.putText(
                    frame_bgr,
                    status,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    color,
                    2, 
                    cv2.LINE_AA,
                    )


            # Show the frame
            cv2.imshow("Phone Detection (just face lol)", frame_bgr)

            # Exit when 'q' is press
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break


    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print ("Webcam released. Programme exited noicely and clean.")



if __name__ == "__main__":
    main()
