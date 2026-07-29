from diffusers import StableDiffusionImg2ImgPipeline
import torch

from configs.config import DEVICE


MODEL_ID = "runwayml/stable-diffusion-v1-5"


class Img2ImgLoader:

    def load(self):

        print("=" * 60)
        print("Loading Image-to-Image Model")
        print("=" * 60)

        dtype = torch.float16 if DEVICE == "cuda" else torch.float32

        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=dtype,
        )

        pipe = pipe.to(DEVICE)

        if DEVICE == "cuda":
            pipe.enable_attention_slicing()

        print("✅ Image-to-Image Model Loaded Successfully.")

        return pipe