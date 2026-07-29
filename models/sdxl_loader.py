from diffusers import StableDiffusionXLPipeline
import torch

from configs.config import DEVICE, MODELS


class SDXLLoader:

    def load(self):

        print("=" * 60)
        print("Loading Stable Diffusion XL...")
        print("=" * 60)

        model_id = MODELS["Stable Diffusion XL"]["id"]

        dtype = torch.float16 if DEVICE == "cuda" else torch.float32

        pipe = StableDiffusionXLPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            use_safetensors=True,
        )

        pipe = pipe.to(DEVICE)

        if DEVICE == "cuda":
            pipe.enable_attention_slicing()

        print("✅ Stable Diffusion XL Loaded Successfully.")

        return pipe