# Python libraries

import time
from pathlib import Path

import cv2
import mediapipe as mp
from ultralytics import YOLO


def _resolve_model_path() -> str:
    """
    Resolve models/fact_detector.model relative to the repo root.
    """

    this_file = Path(__file__).resolve()
    repo_root = this_file.parents[2] # main.py -> phone_detection -> src -> ropo_root
    model_path = repo_root / "models" / "blaze_face_short_range.tflite"

    return str(model_path)


def _draw_face_detection(frame_bgr, detections):
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
                cv2.LINE_AA,)

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



def _draw_phone_boxes(frame_bgr, phone_boxes):
    """
    phone_boxes: list of tubles (x0, y0, x1, y1, conf)
    """

    for (x0, y0, x1, y1, conf) in phone_boxes:
        cv2.rectangle(frame_bgr, (x0, y0), (x1, y1), (255, 255, 0), 2)
        cv2.putText(
                frame_bgr,
                f"Phone {conf:.2f}",
                (x0, max(0, y0 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
                cv2.LINE_AA,)

def _detect_phones_yolo(yolo_model, frame_bgr, conf_thres = 0.5, img_size = 640):
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

    names = r0.names # dict: class_id -> class_name
    if r0.boxes is None or len(r0.boxes) == 0:
        return phone_boxes
    
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
        x0 = int(max(0, min(w - 1, x0)))
        y0 = int(max(0, min(h - 1, y0)))
        x1 = int(max(0, min(w - 1, x1)))
        y1 = int(max(0, min(h - 1, y1)))

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

    ## Face model path
    face_model_path = _resolve_model_path()
    if not Path(face_model_path).exists():
        raise FileNotFoundError(
                f"Could not find model file at:\n {face_model_path}\n\n"
                "Creat <repo>/models and put face_detector inside it.")


    ## Open camera
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

    face_options = FaceDetectorOptions(
            base_options = BaseOptions(model_asset_path = face_model_path),
            running_mode = VisionRunningMode.VIDEO,
            min_detection_confidence = 0.85,
            ) #Change min detection conf to more if want more strict detection





    ## YOLO (phone detection)
    # 'yolov8n.pt' is the smallest + fastest general model.
    # First run will download it,,,, nice
    yolo = YOLO("yolov8n.pt")
    phone_conf = 0.45


    start_ms = int(time.monotonic()*1000)
    last_fps_t = time.monotonic()
    fps = 0.0
    frame_count = 0



    with FaceDetector.create_from_options(face_options) as face_detector:

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
            # Run face detection
            face_result = face_detector.detect_for_video(mp_image, timestamp_ms)

            # Draw face Detections
            if face_result.detections:
                _draw_face_detection(frame_bgr, face_result.detections)

            # Run phone detection
            phone_boxes = _detect_phones_yolo(yolo,
                                              frame_bgr,
                                              conf_thres = phone_conf,
                                              img_size = 640)
            if phone_boxes:
                _draw_phone_boxes(frame_bgr, phone_boxes)


            # New status lines
            n_faces = len(face_result.detections) if face_result.detections else 0
            n_phones = len(phone_boxes)

                #status = f"Faces: {len(result.detections)}"
                #color = (0, 255, 0)
            #else:
                #status = "No face"
                #color = (0, 0, 255)

            if n_faces == 0:
                status = "AWAY (no face)"
                status_color = (0, 0, 255)
            elif n_phones > 0:
                status = "PHONE DETECTED"
                status_color = (0, 255, 255)
            else:
                status = "WORKING (hahahahah)"
                status_color = (0, 255, 0)

            frame_count += 1
            now = time.monotonic()

            if now - last_fps_t >= 1.0:
                fps = frame_count / (now - last_fps_t)
                frame_count = 0
                last_fps_t = now




            cv2.putText(
                    frame_bgr,
                    f"{status} | faces = {n_faces} phones = {n_phones} | fps = {fps:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    status_color,
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
