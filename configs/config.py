MODEL_ID = "runwayml/stable-diffusion-v1-5"

# Device
DEVICE = "cpu"

# Output Folder
OUTPUT_FOLDER = "outputs"

# Image Settings
IMAGE_WIDTH = 512
IMAGE_HEIGHT = 512

# Sampling Settings
NUM_INFERENCE_STEPS = 25
GUIDANCE_SCALE = 7.5

# Default Negative Prompt
NEGATIVE_PROMPT = (
    "blurry, low quality, watermark, text, logo, "
    "distorted, bad anatomy, ugly"
)