# Python libraries

import time
from pathlib import Path

import cv2
import mediapipe as mp

# Local packages
from visualize import (
        draw_face_detection,
        draw_hand_boxes,
        draw_phone_boxes,
        draw_status_line,
        )

from logic import (
        phone_held_by_hand, 
        PhoneUseStateMachine,
        SmoothingConfig,
        )

from detectors.phone import (
        PhoneDetector,
        PhoneDetectorConfig,
        )
from detectors.hand import (
        HandDetector,
        HandDetectorConfig,
        )
from detectors.face import (
        FacePresenceDetector,
        FaceDetectorConfig,
        )

## PATHS to models

def _repo_root() -> Path:
    """
    Assumes: <repo>/src/phone_detection/main.py
    """

    return Path(__file__).resolve().parents[2]

def _model_path(filename: str) -> str:
    """
    Resolve models/model_file  relative to the repo root.

    Face: blaze_face_short_range.tflite
    Hand: hand_landmarker.task
    """

    model_path = _repo_root() / "models" / filename

    return str(model_path)



def main():
    """
    Main programme for phone detection project.

    Scheme:
    - Open face_detector model (Google AI.......);
    - Open the default webcam (maybe a webcam switch function later tho) 
    - Quite function (by pressing some key or a combination of keys)
    """

    ## Paths to models
    # face_model_path = _resolve_model_path()

    face_model = _model_path("blaze_face_short_range.tflite")
    hand_model = _model_path("hand_landmarker.task")
    
    if not Path(face_model).exists():
        raise FileNotFoundError(
                f"Could not find face model file at:\n {face_model}\n\n"
                "Creat <repo>/models and put face_detector inside it.")

    if not Path(hand_model).exists():
        raise FileNotFoundError(
                f"Could not find model file at:\n {hand_model}\n\n"
                "Creat <repo>/models and put hand_detector inside it.")


    ## Open camera
    # Open default camera (usually index 0)
    cap = cv2.VideoCapture(0)
        # Change the index for different camera if there is any, later tho

    # Check if the camera opened successfully 
    if not cap.isOpened():
        raise RuntimeError("ERROR: Could not open camera.")

    print ("Camera opened successfully. Press 'q' to quite.")


    ###### MediaPipe setup
    face_cfg = FaceDetectorConfig(
            model_path = face_model,
            min_detection_confidence = 0.6,
            )

    hand_cfg = HandDetectorConfig(
            model_path = hand_model,
            num_hands = 2,
            min_hand_detection_confidence = 0.5,
            min_hand_presence_confidence = 0.5,
            min_tracking_confidence = 0.5,
            bbox_pad_px = 10,)


    ## YOLO (phone detection)
    phone_detector = PhoneDetector(
            PhoneDetectorConfig(weights = "yolov8n.pt",
                                conf = 0.5,
                                img_size = 640)
            )


    ## Use the state machine (sm) config class instead
    sm = PhoneUseStateMachine(SmoothingConfig(phone_on_frames = 5, 
                                              phone_off_frames = 10,
                                              ))

    # Overlap threshold: (Need fine-tuning)
    HAND_PHONE_IOU = 0.02


    # Timimg
    start_ms = int(time.monotonic()*1000)
    last_fps_t = time.monotonic()
    fps = 0.0
    frame_count = 0



    with FacePresenceDetector(face_cfg) as face_detector,\
            HandDetector(hand_cfg) as hand_detector:


        print ("Running MediaPipe Task Face+Hand Detection. Prease q to quite")


        while True:
            # Read a frame from the webcam
            ret, frame_bgr = cap.read()

            frame_bgr = cv2.flip(frame_bgr, 1) #Flip horizontally

            if not ret:
                print ("WARNING: Failed to grab frame.")
                break

            h, w = frame_bgr.shape[:2]

            # MediaPipe expects RGB
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            # Wrap in mp.Image (SRGB)
            mp_image = mp.Image(image_format = mp.ImageFormat.SRGB, data = frame_rgb)
            # Timestamp required for VIDEO mode (in ms)
            timestamp_ms = int(time.monotonic() * 1000) - start_ms
              


            ###########################################
            ####### Run detection
            #___________________
            # Run face detection
            face_present, face_detections = face_detector.detect(mp_image, timestamp_ms)
#             # Draw face Detections
            if face_detections:
                draw_face_detection(frame_bgr, face_detections)

                
            #____________________
            # Run hand detections

            hand_boxes = hand_detector.detect(mp_image,
                                              timestamp_ms,
                                              image_w = w,
                                              image_h = h,
                                              )

            if hand_boxes:
                draw_hand_boxes(frame_bgr, hand_boxes)

            #____________________
            # Run phone detection
            phone_boxes = phone_detector.detect(frame_bgr)

            if phone_boxes:
                draw_phone_boxes(frame_bgr, phone_boxes)


            #___________________________________
            #__________IF PHONE IN HAND
            phone_held = False
            if hand_boxes and phone_boxes:
                phone_held = phone_held_by_hand(hand_boxes,
                                                phone_boxes,
                                                iou_thres = HAND_PHONE_IOU)

            # Update state machine
            state = sm.update(face_present = face_present, phone_held = phone_held)


            # FPS count
            frame_count += 1
            now = time.monotonic()

            if now - last_fps_t >= 1.0:
                fps = frame_count / (now - last_fps_t)
                frame_count = 0
                last_fps_t = now


            # status show
            if state == "AWAY":
                status_color = (0, 0, 255)
            elif state == "PHONE_USE":
                status_color = (0, 255, 255)
            else:
                status_color = (0, 255, 0)


            debug = f"{state} | F={int(face_present)} H={len(hand_boxes)} P={len(phone_boxes)} PH={int(phone_held)}"

            draw_status_line(frame_bgr = frame_bgr,
                             text = f"{debug} : fps={fps:.1f}",
                             color = status_color,
                             )


            # Show the frame
            cv2.imshow("Phone Detection (Face+Hand+YOLO)", frame_bgr)

            # Exit when 'q' is press
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print ("Webcam released. Programme exited noicely and clean.")



if __name__ == "__main__":
    main()
