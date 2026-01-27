# Python libraries

import cv2
import mediapipe as mp

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
