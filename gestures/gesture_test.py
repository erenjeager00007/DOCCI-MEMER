"""
DOCCI MEMER - Gesture Test

Live webcam test:
webcam -> MediaPipe Face Landmarker -> measurements -> gesture detection

Shows continuously:
- Gesture
- Left EAR
- Right EAR
- Mouth Opening Ratio
- Mouth Width
"""

import cv2
import mediapipe as mp

from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    FaceLandmarker,
    FaceLandmarkerOptions,
    RunningMode,
)

from gestures.gesture_detector import detect_gesture
from gestures.gesture_stabilizer import GestureStabilizer
from gestures.facial_measurements import (
    left_eye_ear,
    right_eye_ear,
    mouth_opening_ratio,
    mouth_width,
)

MODEL_PATH = "models/face_landmarker.task"
WINDOW_NAME = "DOCCI MEMER - Gesture Test"


def draw_landmarks(frame, face_landmarks):
    height, width = frame.shape[:2]

    for landmark in face_landmarks:
        x = int(landmark.x * width)
        y = int(landmark.y * height)

        cv2.circle(
            frame,
            (x, y),
            1,
            (0, 255, 0),
            -1
        )


def draw_text(frame, text, x, y, size=0.6):
    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        size,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def main():

    stabilizer = GestureStabilizer(stable_frames=5)

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=MODEL_PATH
        ),
        running_mode=RunningMode.VIDEO,
        num_faces=1,
    )

    landmarker = FaceLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        landmarker.close()
        return

    cv2.namedWindow(
        WINDOW_NAME,
        cv2.WINDOW_NORMAL
    )

    print("Webcam and gesture detector ready.")
    print("Press Q or ESC to quit.")

    frame_timestamp_ms = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to grab frame.")
            break

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        result = landmarker.detect_for_video(
            mp_image,
            frame_timestamp_ms
        )

        frame_timestamp_ms += 1

        gesture = "NO FACE"
        left_ear = 0.0
        right_ear = 0.0
        mouth_ratio = 0.0
        width = 0.0

        if result.face_landmarks:

            face_landmarks = result.face_landmarks[0]

            draw_landmarks(
                frame,
                face_landmarks
            )

            left_ear = left_eye_ear(face_landmarks)
            right_ear = right_eye_ear(face_landmarks)
            mouth_ratio = mouth_opening_ratio(face_landmarks)
            width = mouth_width(face_landmarks)

            detected_gesture = detect_gesture(
                face_landmarks
            )

            gesture = stabilizer.update(
                detected_gesture
            )

        draw_text(
            frame,
            f"Gesture: {gesture}",
            20,
            40,
            1.0
        )

        draw_text(
            frame,
            f"Left EAR: {left_ear:.3f}",
            20,
            80
        )

        draw_text(
            frame,
            f"Right EAR: {right_ear:.3f}",
            20,
            110
        )

        draw_text(
            frame,
            f"Mouth Opening Ratio: {mouth_ratio:.3f}",
            20,
            140
        )

        draw_text(
            frame,
            f"Mouth Width: {width:.3f}",
            20,
            170
        )

        cv2.imshow(
            WINDOW_NAME,
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == ord("Q") or key == 27:
            break

        if cv2.getWindowProperty(
            WINDOW_NAME,
            cv2.WND_PROP_VISIBLE
        ) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)

    landmarker.close()


if __name__ == "__main__":
    main()