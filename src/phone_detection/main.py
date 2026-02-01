# Python libraries

import time
from pathlib import Path

import cv2
import mediapipe as mp
from ultralytics import YOLO

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
        intersects,
        iou,
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

## Define Geometry stuff

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

def _get_face_boxes(detections):
    """
    Return list of face boxes as (x0, y0, x1, y1) in pixel coordinates.
    """

    boxes = []
    for det in detections or []:
        bb = det.bounding_box
        x0 = int(bb.origin_x)
        y0 = int(bb.origin_y)
        x1 = int(bb.origin_x + bb.width)
        y1 = int(bb.origin_y + bb.height)
        boxes.append((x0, y0, x1, y1))

    return boxes


## YOLO PHone detectionj

def _detect_phones_yolo(yolo_model, frame_bgr, conf_thres = 0.4, img_size = 640):
    """
    Run YOLO and return phone boxes in original frame coordinates. 
    Returns: list of (x0, y0, x1, y1, conf.)
    """

    h, w = frame_bgr.shape[:2]

    # Resize for speed (keep aspect ratio by simple scaling)
    scale = img_size/max(h, w)

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
    results = yolo_model.predict(small, conf = conf_thres, verbose=False)
    
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
    BaseOptions = mp.tasks.BaseOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    FaceDetector = mp.tasks.vision.FaceDetector
    FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions

    face_options = FaceDetectorOptions(
            base_options = BaseOptions(model_asset_path = face_model),
            running_mode = VisionRunningMode.VIDEO,
            min_detection_confidence = 0.85,
            ) #Change min detection conf to more if want more strict detection

    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

    hand_options = HandLandmarkerOptions(
            base_options = BaseOptions(model_asset_path = hand_model),
            running_mode = VisionRunningMode.VIDEO,
            num_hands = 2,
            min_hand_detection_confidence = 0.5,
            min_hand_presence_confidence = 0.5,
            min_tracking_confidence = 0.5, 
            )





    ## YOLO (phone detection)
    # 'yolov8n.pt' is the smallest + fastest general model.
    # First run will download it,,,, nice
    yolo = YOLO("yolov8n.pt")
    phone_conf = 0.4

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



    with FaceDetector.create_from_options(face_options) as face_detector,\
            HandLandmarker.create_from_options(hand_options) as hand_landmarker:


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
            face_result = face_detector.detect_for_video(mp_image, timestamp_ms)
            face_present = bool(face_result.detections)
            # Draw face Detections
            if face_result.detections:
                draw_face_detection(frame_bgr, face_result.detections)
                
            #____________________
            # Run hand detections
            hand_result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)
            # Draw hand boxex
            hand_boxes  = []
            if hand_result.hand_landmarks:
                for hand_lms in hand_result.hand_landmarks:
                    hb = _bbox_from_normalized_landmarks(hand_lms,
                                                         w = w, 
                                                         h = h,
                                                         pad = 10)
                    if hb is not None:
                        hand_boxes.append(hb)

            if hand_boxes:
                draw_hand_boxes(frame_bgr, hand_boxes)

            #____________________
            # Run phone detection
            phone_boxes = _detect_phones_yolo(yolo,
                                              frame_bgr,
                                              conf_thres = phone_conf,
                                              img_size = 640)
            if phone_boxes:
                draw_phone_boxes(frame_bgr, phone_boxes)



            #___________________________________
            #__________IF PHONE IN HAND
            phone_held = False
            if hand_boxes and phone_boxes:
                phone_held = phone_held_by_hand(hand_boxes,
                                                phone_boxes,
                                                iou_thres = HAND_PHONE_IOU)


            state = sm.update(face_present = face_present, phone_held = phone_held)



            # New status lines
            n_faces = len(face_result.detections) if face_result.detections else 0
            n_phones = len(phone_boxes)

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
