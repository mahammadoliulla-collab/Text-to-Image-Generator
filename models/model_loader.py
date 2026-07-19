from diffusers import StableDiffusionPipeline
import torch

from configs.config import MODEL_ID
from configs.config import DEVICE


class ModelLoader:

    def load(self):

        print("Loading model...")

        pipe = StableDiffusionPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float32
        )

        pipe = pipe.to(DEVICE)

        print("Model Loaded Successfully.")

        return pipe