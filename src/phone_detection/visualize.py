import cv2

def _clamp(v, lo, hi):
    return max(lo, min(hi, v))

def draw_face_detection(frame_bgr, detections):
    """
    Draw face bounding boxes + confidence on the frame
    """

    h, w  = frame_bgr.shape[:2]

    for det in detections:
        # det.bounding_box is basically the input image coordinations.
        bbox = det.bounding_box
        x0 = _clamp(int(bbox.origin_x), 0, w - 1)
        y0 = _clamp(int(bbox.origin_y), 0, h - 1)
        x1 = _clamp(int(bbox.origin_x + bbox.width), 0, w - 1)
        y1 = _clamp(int(bbox.origin_y + bbox.height), 0, h - 1)

        cv2.rectangle(frame_bgr, (x0, y0), (x1, y1), (0, 255, 0), 2)


        # Confidence score is usually in categories[0].score

        score = det.categories[0].score if det.categories else None

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

def draw_hand_boxes(frame_bgr, hand_boxes):
    for (x0, y0, x1, y1) in hand_boxes:
        cv2.rectangle(frame_bgr, (x0, y0), (x1, y1), (255, 0, 255), 2)
        cv2.putText(
                frame_bgr,
                "Hand",
                (x0, max(0, y0 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 255),
                2, 
                cv2.LINE_AA,
                )

def draw_phone_boxes(frame_bgr, phone_boxes):
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
                cv2.LINE_AA,
                )

def draw_status_line(frame_bgr, text, color = (255, 255, 255), origin = (10, 30)):

    """
    Draw a single status line at the (10, 30), top-left ish
    """

    cv2.putText(
            frame_bgr,
            text,
            origin,
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7,
            color,
            2, 
            cv2.LINE_AA,
            )
