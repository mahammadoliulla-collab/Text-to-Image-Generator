import json
import os

HISTORY_FILE = "outputs/history.json"


def save_history(record):

    history = []

    if os.path.exists(HISTORY_FILE):

        try:

            with open(HISTORY_FILE, "r") as f:

                history = json.load(f)

        except:

            history = []

    history.insert(0, record)

    history = history[:20]

    with open(HISTORY_FILE, "w") as f:

        json.dump(history, f, indent=4)


def load_history():

    if not os.path.exists(HISTORY_FILE):

        return []

    try:

        with open(HISTORY_FILE, "r") as f:

            return json.load(f)

    except:

        return []