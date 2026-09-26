"""
DOCCI MEMER - Facial Measurements

Reusable helper functions that turn MediaPipe Face Landmarker points
into simple numeric measurements (Eye Aspect Ratio, mouth opening, etc.).

These numbers are the building blocks for gesture detection later
(e.g. "eyes closed" or "mouth open"), but this file only calculates
the numbers - it does not detect gestures itself.
"""

import numpy as np

# ---------------------------------------------------------------------
# Landmark indices (MediaPipe Face Landmarker / Face Mesh, 468 points)
# ---------------------------------------------------------------------

# Right eye (the person's own right eye, appears on the LEFT side of a
# mirrored webcam image). 6 points used for the classic EAR formula:
# [outer corner, top-1, top-2, inner corner, bottom-1, bottom-2]
RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144]

# Left eye (the person's own left eye, appears on the RIGHT side of a
# mirrored webcam image).
LEFT_EYE_IDX = [362, 385, 387, 263, 373, 380]

# Mouth points
MOUTH_LEFT_CORNER_IDX = 61
MOUTH_RIGHT_CORNER_IDX = 291
MOUTH_UPPER_LIP_IDX = 13   # inner top lip, center
MOUTH_LOWER_LIP_IDX = 14   # inner bottom lip, center


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def _get_point(face_landmarks, index):
    """
    Safely get a landmark's (x, y) position as a NumPy array.

    Returns None if the index is out of range or the landmark list
    is empty/invalid, instead of crashing the program.
    """
    if face_landmarks is None:
        return None

    if index < 0 or index >= len(face_landmarks):
        return None

    landmark = face_landmarks[index]

    if landmark is None:
        return None

    return np.array([landmark.x, landmark.y])


def _distance(point_a, point_b):
    """Euclidean distance between two 2D points."""
    if point_a is None or point_b is None:
        return 0.0

    return float(np.linalg.norm(point_a - point_b))


# ---------------------------------------------------------------------
# Eye Aspect Ratio (EAR)
# ---------------------------------------------------------------------
#
# EAR measures how "open" an eye is, using 6 points around it:
#
#       p2   p3
#     p1        p4
#       p6   p5
#
# EAR = (distance(p2,p6) + distance(p3,p5)) / (2 * distance(p1,p4))
#
# A wide-open eye gives a higher EAR. A closed/blinking eye gives a
# low EAR (close to 0), because the top and bottom points move close
# together while the eye stays the same width.

def _calculate_ear(face_landmarks, eye_indices):
    """Calculate the Eye Aspect Ratio for one eye given its 6 indices."""
    points = [_get_point(face_landmarks, idx) for idx in eye_indices]

    # If any point is missing, we can't safely calculate EAR.
    if any(point is None for point in points):
        return 0.0

    p1, p2, p3, p4, p5, p6 = points

    vertical_1 = _distance(p2, p6)
    vertical_2 = _distance(p3, p5)
    horizontal = _distance(p1, p4)

    # Avoid dividing by zero if landmarks are degenerate/overlapping.
    if horizontal == 0:
        return 0.0

    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return float(ear)


def left_eye_ear(face_landmarks):
    """Eye Aspect Ratio for the person's left eye."""
    return _calculate_ear(face_landmarks, LEFT_EYE_IDX)


def right_eye_ear(face_landmarks):
    """Eye Aspect Ratio for the person's right eye."""
    return _calculate_ear(face_landmarks, RIGHT_EYE_IDX)


def average_ear(face_landmarks):
    """
    Average Eye Aspect Ratio across both eyes.

    Useful because both eyes usually open/close together, so averaging
    reduces noise from a single eye's measurement.
    """
    left = left_eye_ear(face_landmarks)
    right = right_eye_ear(face_landmarks)
    return float((left + right) / 2.0)


# ---------------------------------------------------------------------
# Mouth measurements
# ---------------------------------------------------------------------

def mouth_opening_ratio(face_landmarks):
    """
    How open the mouth is, relative to mouth width.

    Calculated as: vertical lip gap / horizontal mouth width.
    A closed mouth gives a value near 0. An open mouth gives a
    larger value. Dividing by width keeps the ratio consistent
    whether the face is close to or far from the camera.
    """
    top_lip = _get_point(face_landmarks, MOUTH_UPPER_LIP_IDX)
    bottom_lip = _get_point(face_landmarks, MOUTH_LOWER_LIP_IDX)
    left_corner = _get_point(face_landmarks, MOUTH_LEFT_CORNER_IDX)
    right_corner = _get_point(face_landmarks, MOUTH_RIGHT_CORNER_IDX)

    if top_lip is None or bottom_lip is None or left_corner is None or right_corner is None:
        return 0.0

    vertical_gap = _distance(top_lip, bottom_lip)
    width = _distance(left_corner, right_corner)

    if width == 0:
        return 0.0

    ratio = vertical_gap / width
    return float(ratio)


def mouth_width(face_landmarks):
    """
    Horizontal distance between the two mouth corners.

    This is a raw (unnormalized) distance in the same coordinate
    scale as the landmarks themselves - useful for comparing mouth
    width changes, e.g. smiling stretches the mouth wider.
    """
    left_corner = _get_point(face_landmarks, MOUTH_LEFT_CORNER_IDX)
    right_corner = _get_point(face_landmarks, MOUTH_RIGHT_CORNER_IDX)

    return _distance(left_corner, right_corner)