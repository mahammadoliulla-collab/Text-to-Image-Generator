import json
import os


HISTORY_FILE = "history.json"


def show_history():

    if not os.path.exists(HISTORY_FILE):
        print("\nNo generation history found.")
        return

    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as file:
            history = json.load(file)

    except:
        print("\nHistory file is corrupted.")
        return

    if len(history) == 0:
        print("\nHistory is empty.")
        return

    print("\n" + "=" * 60)
    print("        IMAGE GENERATION HISTORY")
    print("=" * 60)

    for index, item in enumerate(history, start=1):

        print(f"\nImage #{index}")
        print("-" * 60)

        print(f"Prompt           : {item['prompt']}")
        print(f"Image            : {item['image']}")
        print(f"Seed             : {item['seed']}")
        print(f"Width            : {item['width']}")
        print(f"Height           : {item['height']}")
        print(f"Steps            : {item['steps']}")
        print(f"CFG Scale        : {item['cfg']}")
        print(f"Generated At     : {item['generated_at']}")

    print("\n" + "=" * 60)
    print(f"Total Images Generated : {len(history)}")
    print("=" * 60)