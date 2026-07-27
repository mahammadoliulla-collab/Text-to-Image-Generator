from models.model_loader import ModelLoader
from src.generator import ImageGenerator

import glob
import os
from PIL import Image

# ==========================================
# USER SETTINGS
# Change these values whenever you want
# ==========================================

PROMPT = "A futuristic cyberpunk city at sunset"

NEGATIVE_PROMPT = ""

WIDTH = 512
HEIGHT = 512

STEPS = 25
CFG = 7.5

SEED = 42

NUM_IMAGES = 1

# ==========================================
# DO NOT EDIT BELOW THIS LINE
# ==========================================

print("=" * 60)
print("TEXT TO IMAGE GENERATOR (GOOGLE COLAB)")
print("=" * 60)

loader = ModelLoader()

pipe = loader.load()

generator = ImageGenerator(pipe)

generator.generate(
    PROMPT,
    NEGATIVE_PROMPT,
    WIDTH,
    HEIGHT,
    STEPS,
    CFG,
    SEED,
    NUM_IMAGES,
)

# ==========================================
# Display latest generated image
# ==========================================

images = glob.glob("outputs/*.png")

if images:

    latest_image = max(images, key=os.path.getctime)

    print("\n" + "=" * 60)
    print("LATEST GENERATED IMAGE")
    print("=" * 60)
    print(f"Image Path : {latest_image}")

    try:
        from IPython.display import display
        display(Image.open(latest_image))
    except ImportError:
        print("Image generated successfully.")
        print(f"Saved at: {latest_image}")

else:
    print("No generated image found.")