import cv2

def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    window_name = "DOGGO MEMER - Camera Test"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("Webcam opened successfully. Press 'Q' or 'ESC' to quit.")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to grab frame.")
            break

        cv2.imshow(window_name, frame)

        # Bring focus to the window and check for key press
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == ord('Q') or key == 27:  # 27 = ESC
            break

        # If the window was closed via the 'X' button, exit too
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)  # ensures window actually closes on some systems

if __name__ == "__main__":
    main()