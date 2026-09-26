from gestures.facial_measurements import (
    left_eye_ear,
    right_eye_ear,
    mouth_opening_ratio,
)

# Eye thresholds
EYE_CLOSED_EAR_THRESHOLD = 0.23
EYE_OPEN_EAR_THRESHOLD = 0.27
WINK_EAR_DIFFERENCE_THRESHOLD = 0.05

# Mouth threshold
MOUTH_OPEN_RATIO_THRESHOLD = 0.45

# Smile detection
SMILE_MOUTH_RATIO_THRESHOLD = 0.08


GESTURE_NEUTRAL = "NEUTRAL"
GESTURE_SMILE = "SMILE"
GESTURE_WINK = "WINK"
GESTURE_OPEN_MOUTH = "OPEN_MOUTH"


def detect_gesture(face_landmarks):

    if not face_landmarks:
        return GESTURE_NEUTRAL

    left_ear = left_eye_ear(face_landmarks)
    right_ear = right_eye_ear(face_landmarks)
    mouth_ratio = mouth_opening_ratio(face_landmarks)

    if left_ear == 0.0 and right_ear == 0.0:
        return GESTURE_NEUTRAL

    # Calculate difference between eyes
    ear_difference = abs(left_ear - right_ear)

    # -------------------------
    # WINK
    # -------------------------

    left_closed = left_ear < EYE_CLOSED_EAR_THRESHOLD
    right_closed = right_ear < EYE_CLOSED_EAR_THRESHOLD

    left_open = left_ear > EYE_OPEN_EAR_THRESHOLD
    right_open = right_ear > EYE_OPEN_EAR_THRESHOLD

    if (
        (left_closed and right_open)
        or
        (right_closed and left_open)
    ) and ear_difference > WINK_EAR_DIFFERENCE_THRESHOLD:

        return GESTURE_WINK

    # -------------------------
    # OPEN MOUTH
    # -------------------------

    if mouth_ratio > MOUTH_OPEN_RATIO_THRESHOLD:
        return GESTURE_OPEN_MOUTH

    # -------------------------
    # SMILE
    # -------------------------

    if mouth_ratio > SMILE_MOUTH_RATIO_THRESHOLD:
        return GESTURE_SMILE

    # -------------------------
    # NEUTRAL
    # -------------------------

    return GESTURE_NEUTRAL