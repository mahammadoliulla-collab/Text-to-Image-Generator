import torch

# ==========================================================
# AVAILABLE MODELS
# ==========================================================

MODELS = {

    "Stable Diffusion 1.5": {
        "id": "runwayml/stable-diffusion-v1-5",
        "type": "sd15",
        "width": 512,
        "height": 512,
        "steps": 25,
        "cfg": 7.5,
    },

    "Stable Diffusion XL": {
        "id": "stabilityai/stable-diffusion-xl-base-1.0",
        "type": "sdxl",
        "width": 1024,
        "height": 1024,
        "steps": 30,
        "cfg": 7.0,
    },

    # Uncomment later
    # "FLUX.1 Dev": {
    #     "id": "black-forest-labs/FLUX.1-dev",
    #     "type": "flux",
    #     "width": 1024,
    #     "height": 1024,
    #     "steps": 28,
    #     "cfg": 3.5,
    # },

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
# DEFAULT IMAGE SETTINGS
# (Used only as fallback)
# ==========================================================

IMAGE_WIDTH = 512
IMAGE_HEIGHT = 512

# ==========================================================
# DEFAULT SAMPLING
# (Used only as fallback)
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