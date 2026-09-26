"""
DOCCI MEMER - Face Landmark Detection Test

Detects a single face from the webcam using MediaPipe Face Landmarker
and draws the landmarks on the video frame.
"""

import cv2
import mediapipe as mp

from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    FaceLandmarker,
    FaceLandmarkerOptions,
    RunningMode,
)

MODEL_PATH = "models/face_landmarker.task"
WINDOW_NAME = "DOCCI MEMER - Face Detector Test"


def draw_landmarks(frame, face_landmarks):
    """Draw each detected landmark as a small dot on the frame."""
    height, width = frame.shape[:2]

    for landmark in face_landmarks:
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)


def main():
    # Set up the Face Landmarker in VIDEO mode (good fit for a webcam loop)
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        num_faces=1,
    )

    landmarker = FaceLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        landmarker.close()
        return

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    print("Webcam and face detector ready. Press 'Q' or 'ESC' to quit.")

    frame_timestamp_ms = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to grab frame.")
            break

        # Convert the BGR OpenCV frame to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Run detection for this frame (timestamp must increase each call)
        result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
        frame_timestamp_ms += 1

        # Draw landmarks if a face was found
        if result.face_landmarks:
            draw_landmarks(frame, result.face_landmarks[0])

        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q') or key == 27:  # 27 = ESC
            break

        # If the window was closed via the 'X' button, exit too
        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    landmarker.close()


if __name__ == "__main__":
    main()