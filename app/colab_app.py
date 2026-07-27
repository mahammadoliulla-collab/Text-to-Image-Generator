from models.model_loader import ModelLoader
from src.generator import ImageGenerator

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