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
        x1 = int(bbox.origion_x + bbox.width)
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
                (0, 255, 0)
                2, 
                cv2.LINE_AA,
                )










def main():
    """
    Main programme for phone detection project.

    Scheme:
    1. Open the default webcam (maybe a webcam switch function later tho)
    2. Show live feed video (also maybe just a sound warning and show funny memes instead)
    3. Quite function (by pressing some key or a combination of keys)


    """


    # Open default camera (usually index 0)
    cap = cv2.VideoCapture(0)
        # Change the index for different camera if there is any, later tho

    # Check if the camera opened successfully 
    if not cap.isOpened():
        raise RuntimeError("ERROR: Could not open camera.")

    print ("Camera opened successfully. Press 'q' to quite.")



    # MediaPipe setup
    mp_face = mp.solutions.face_detection
    mp_draw = mp.solutions.drawing_utils

    # Model_selection:
    # 0 = short-range (~ within 2m, for laptop used)
    # 1 = full-range (works better for far distances)

    with mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.6) as face_detector:
        print ("Running face detection")


        while True:
            # Read a frame from the webcam
            ret, frame = cap.read()

            if not ret:
                print ("WARNING: Failed to grab frame.")
                break

            # MediaPipe expects RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Run face detection
            results = face_detector.process(frame_rgb)

            # Draw Detections
            if results.detections:
                for det in results.detections:
                    # Draw bounding box + keypoints (eyes, nose, mouth) using MediaPipe helper
                    mp_draw.draw_detection(frame, det)

                    # Show how confidence for the detections
                    score = det.score[0] if det.score else 0.0
                    cv2.putText(
                            frame,
                            f"Face: {score:.2f}",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.0,
                            (0, 255, 0),
                            2,
                            cv2.LINE_AA,
                            )
                else:
                    cv2.putText(
                            frame,
                            "NO FACE",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.0,
                            (0, 0, 255),
                            2,
                            cv2.LINE_AA,
                            )


            # Show the frame
            cv2.imshow("Phone Detection", frame)

            # Exit when 'q' is press
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break


    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print ("Webcam released. Programme exited noicely and clean.")



if __name__ == "__main__":
    main()
