# meme_mapper.py
# Maps facial gesture names to meme categories.

# Dictionary that maps each known gesture to its meme category
GESTURE_TO_MEME = {
    "NEUTRAL": "neutral",
    "SMILE": "happy",
    "WINK": "wink",
    "OPEN_MOUTH": "shocked",
}


def get_meme_category(gesture):
    """
    Given a gesture name (like "SMILE"), return the matching meme category.
    If the gesture is not recognized, return "neutral" as a safe default.
    """
    return GESTURE_TO_MEME.get(gesture, "neutral")


# Simple test area - runs only if this file is executed directly
if __name__ == "__main__":
    test_gestures = ["NEUTRAL", "SMILE", "WINK", "OPEN_MOUTH", "UNKNOWN_GESTURE"]

    for gesture in test_gestures:
        category = get_meme_category(gesture)
        print(f"{gesture} -> {category}")