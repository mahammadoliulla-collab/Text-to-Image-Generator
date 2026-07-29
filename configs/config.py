import torch

# ==========================================================
# MODEL CONFIGURATION
# ==========================================================

MODELS = {
    "Stable Diffusion 1.5": "runwayml/stable-diffusion-v1-5",
    #"SDXL": "stabilityai/stable-diffusion-xl-base-1.0",
    #"FLUX.1 Dev": "black-forest-labs/FLUX.1-dev",
}

DEFAULT_MODEL = "Stable Diffusion 1.5"

# ==========================================================
# DEVICE
# ==========================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================================
# OUTPUT
# ==========================================================

OUTPUT_FOLDER = "outputs"

# ==========================================================
# IMAGE SETTINGS
# ==========================================================

IMAGE_WIDTH = 512
IMAGE_HEIGHT = 512

# ==========================================================
# SAMPLING
# ==========================================================

NUM_INFERENCE_STEPS = 25
GUIDANCE_SCALE = 7.5

# ==========================================================
# NEGATIVE PROMPT
# ==========================================================

NEGATIVE_PROMPT = (
    "blurry, low quality, watermark, text, logo, "
    "distorted, bad anatomy, ugly"
)