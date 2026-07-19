import json
import os


HISTORY_FILE = "history.json"


def save_history(entry):
    """
    Save every generated image information
    into history.json
    """

    history = []

    # Load previous history
    if os.path.exists(HISTORY_FILE):

        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as file:
                history = json.load(file)

        except:
            history = []

    # Add new entry
    history.append(entry)

    # Save updated history
    with open(HISTORY_FILE, "w", encoding="utf-8") as file:
        json.dump(
            history,
            file,
            indent=4,
            ensure_ascii=False,
        )