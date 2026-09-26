# meme_loader.py
# Loads meme image file paths from folders, organized by category.

from pathlib import Path

# The image file extensions we care about
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp"]


class MemeLoader:
    """
    A simple class that finds meme image file paths
    inside a category folder, e.g. dog_memes/happy/
    """

    def __init__(self, base_folder="dog_memes"):
        self.base_folder = Path(base_folder)

    def get_memes(self, category):
        """
        Look inside base_folder/category/ and return a list of
        file paths for any images found there.

        If the folder doesn't exist or has no images,
        return an empty list.
        """
        category_folder = self.base_folder / category

        if not category_folder.exists():
            return []

        memes = []
        for file_path in category_folder.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS:
                memes.append(file_path)

        return memes


# Simple test area - runs only if this file is executed directly
if __name__ == "__main__":
    loader = MemeLoader()

    test_categories = ["happy", "shocked", "wink", "neutral"]

    for category in test_categories:
        memes = loader.get_memes(category)
        print(f"{category}: {memes}")