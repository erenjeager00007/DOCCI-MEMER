# reaction_manager.py
# Manages reactions by mapping gestures to their meme category.

from reactions.meme_mapper import get_meme_category


class ReactionManager:
    """
    A simple class that takes a gesture and figures out
    which meme category it belongs to.
    """

    def get_reaction(self, gesture):
        """
        Given a gesture name, return a dictionary with:
        - the original gesture
        - the matching meme category
        """
        category = get_meme_category(gesture)

        return {
            "gesture": gesture,
            "category": category,
        }


# Simple test area - runs only if this file is executed directly
if __name__ == "__main__":
    manager = ReactionManager()

    test_gestures = ["NEUTRAL", "SMILE", "WINK", "OPEN_MOUTH", "UNKNOWN_GESTURE"]

    for gesture in test_gestures:
        result = manager.get_reaction(gesture)
        print(result)