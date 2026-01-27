# Python libraries
import cv2

def main():
    """
    Main programme for phone detection project.

    Scheme:
    1. Open the default webcam (maybe a webcam switch function later tho)
    2. Show live feed video (also maybe just a sound warning and show funny memes instead)
    3. Quite function (by pressing some key or a combination of keys)


    """


    # Open default camera (usually index 0)
    cap = cv2.VideoCapture(0)  # Change the index for different camera if there is any, later tho

    # Check if the camera opened successfully 
    if not cap.isOpened():
        raise RuntimeError("ERROR: Could not open camera.")

    print ("Camera opened successfully. Press 'q' to quite.")


    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()

        if not ret:
            print ("WARNING: Failed to grab frame.")
            break


        # Show the frame
        cv2.imshow("Phone Detection", frame)

        # Exit when 'q' is press
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print ("Webcam released. Programme exited noicely and clean.")



if __name__ = "__main__":
    main()
