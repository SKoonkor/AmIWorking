# Python libraries

import time
from pathlib import Path

import cv2
import mediapipe as mp

# Local packages
from phone_detection.visualize import (
        draw_face_detection,
        draw_hand_boxes,
        draw_phone_boxes,
        draw_status_line,
        )

from phone_detection.logic import (
        phone_held_by_hand, 
        PhoneUseStateMachine,
        SmoothingConfig,
        )

from phone_detection.detectors.phone import (
        PhoneDetector,
        PhoneDetectorConfig,
        )
from phone_detection.detectors.hand import (
        HandDetector,
        HandDetectorConfig,
        )
from phone_detection.detectors.face import (
        FacePresenceDetector,
        FaceDetectorConfig,
        )

from phone_detection.camera import (
        Camera,
        CameraConfig,
        )

from phone_detection.config import load_settings, resolve_model_path

## PATHS to models

def main():
    """
    Main programme for phone detection project.

    Scheme:
    - Open face_detector model (Google AI.......);
    - Open the default webcam (maybe a webcam switch function later tho) 
    - Quite function (by pressing some key or a combination of keys)
    """

    ## Open setting from config/settings.toml
    settings = load_settings()

    phone_grace_s = settings.tracking.get("phone_grace_s", 0.7) # Improving phone and hand detection
    hand_grace_s = settings.tracking.get("hand_grace_s", 0.7)


    ## Open camera
    resize_max = settings.camera.get("resize_max", None)
    if resize_max == 0:
        resize_max = None

    camera_cfg = CameraConfig(
            index = settings.camera.get("index", 0),
            flip_horizontal = settings.camera.get("flip_horizontal", True),
            resize_max = resize_max,)

    ###### MediaPipe setup
    ## Paths to models
    face_model = resolve_model_path(settings.models["face_task"])
    hand_model = resolve_model_path(settings.models["hand_task"])
    yolo_weights = settings.models.get("yolo_weights", "yolov8n.pt")

    # detectors cfg
    face_cfg = FaceDetectorConfig(
            model_path = face_model,
            min_detection_confidence = settings.face.get("min_detection_confidence", 0.6),
            )

    hand_cfg = HandDetectorConfig(
            model_path = hand_model,
            num_hands = settings.hands.get("num_hands", 2),
            min_hand_detection_confidence = settings.hands.get("min_hand_detection_confidence", 0.5),
            min_hand_presence_confidence = settings.hands.get("min_hand_presence_confidence", 0.5),
            min_tracking_confidence = settings.hands.get("min_tracking_confidence", 0.5),
            bbox_pad_px = settings.hands.get("bbox_pad_px", 10),
            ) # the default will be tighen up using HandDetectorConfig(..., **settings.hands)


    ## YOLO (phone detection)
    phone_detector = PhoneDetector(
            PhoneDetectorConfig(
                weights = yolo_weights,
                conf = settings.phone.get("conf", 0.5),
                img_size = settings.phone.get("img_size", 640),
            )
            )

    ## Use the state machine (sm) config class instead
    sm = PhoneUseStateMachine(
        SmoothingConfig(
            phone_on_frames = settings.smoothing.get("phone_on_frames", 5),
            phone_off_frames = settings.smoothing.get("phone_off_frames", 10),
            )
        )

                

    # Overlap threshold: (Need fine-tuning)
    HAND_PHONE_IOU = settings.association.get("hand_phone_iou", 0.02)
    REQUIRE_IOU = settings.association.get("require_iou", True)


    # Timimg
    start_ms = int(time.monotonic()*1000)
    last_fps_t = time.monotonic()
    fps = 0.0
    frame_count = 0
    last_phone_seen_t = -1e9
    last_hand_seen_t = -1e9



    with Camera(camera_cfg) as cam,\
            FacePresenceDetector(face_cfg) as face_detector,\
            HandDetector(hand_cfg) as hand_detector:


        print ("Running MediaPipe Task Face+Hand Detection. Prease q to quite")


        while True:
            # Read a frame from the webcam
            #ret, frame_bgr = cap.read()
            frame_bgr = cam.read()


            if frame_bgr is None:
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
            now_t = time.monotonic() # Setting time to once hand/phone detected 
            hand_seen_now = len(hand_boxes) > 0
            if hand_seen_now:
                last_hand_seen_t = now_t

            if hand_boxes:
                draw_hand_boxes(frame_bgr, hand_boxes)


            #____________________
            # Run phone detection
            phone_boxes = phone_detector.detect(frame_bgr)

            phone_seen_now = len(phone_boxes) > 0
            if phone_seen_now:
                last_phone_seen_t = now_t
            if phone_boxes:
                draw_phone_boxes(frame_bgr, phone_boxes)


            # Phone and Hand appreared on screen recently?
            hand_recent = hand_seen_now or (now_t - last_hand_seen_t) < hand_grace_s
            phone_recent = phone_seen_now or (now_t - last_phone_seen_t) < phone_grace_s

              


            #___________________________________
            #__________IF PHONE IN HAND
            phone_held = False
#             if hand_boxes and phone_boxes:
#                 phone_held = phone_held_by_hand(hand_boxes,
#                                                 phone_boxes,
#                                                 iou_thres = HAND_PHONE_IOU,
#                                                 require_iou = REQUIRE_IOU,)
            # New logic for holding phone based on phone and hand both detected now
            if hand_seen_now and phone_seen_now:
                phone_held = phone_held_by_hand(
                        hand_boxes,
                        phone_boxes,
                        iou_thres = HAND_PHONE_IOU,
                        require_iou = REQUIRE_IOU,
                        )

            elif phone_seen_now and hand_recent and sm.state == "PHONE_USE":
                # Recently saw a hand. Assume it is still holding during short occlusion.
                phone_held = True

            elif hand_seen_now and phone_recent:
                # Recently saw a phone. Assume it is still being held.
                phone_held = True

            

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
    cv2.destroyAllWindows()
    print ("\n\n\nWebcam released. Programme exited noicely and clean.")



if __name__ == "__main__":
    main()
