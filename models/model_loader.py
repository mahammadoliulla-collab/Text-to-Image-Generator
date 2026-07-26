from diffusers import StableDiffusionPipeline
import torch

from configs.config import MODEL_ID
from configs.config import DEVICE


class ModelLoader:

    def load(self):

        print("Loading model...")

        # Use float16 on GPU, float32 on CPU
        dtype = torch.float16 if DEVICE == "cuda" else torch.float32

        pipe = StableDiffusionPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=dtype
        )

        pipe = pipe.to(DEVICE)

        # Optional optimization for NVIDIA GPUs
        if DEVICE == "cuda":
            pipe.enable_attention_slicing()

        print("Model Loaded Successfully.")

        return pipe