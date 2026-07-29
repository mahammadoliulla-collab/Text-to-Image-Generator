from diffusers import StableDiffusionPipeline
import torch

from configs.config import DEVICE, MODELS
from models.sdxl_loader import SDXLLoader


class ModelLoader:

    def load(self, model_name):

        model_info = MODELS[model_name]

        model_type = model_info["type"]

        # ======================================================
        # SDXL
        # ======================================================

        if model_type == "sdxl":

            return SDXLLoader().load()

        # ======================================================
        # Stable Diffusion 1.5
        # ======================================================

        print("=" * 60)
        print(f"Loading {model_name}...")
        print("=" * 60)

        dtype = torch.float16 if DEVICE == "cuda" else torch.float32

        pipe = StableDiffusionPipeline.from_pretrained(
            model_info["id"],
            torch_dtype=dtype,
        )

        pipe = pipe.to(DEVICE)

        if DEVICE == "cuda":
            pipe.enable_attention_slicing()

        print(f"✅ {model_name} Loaded Successfully.")

        return pipe