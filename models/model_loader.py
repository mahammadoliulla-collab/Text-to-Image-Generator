from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
)

import torch

from configs.config import (
    MODELS,
    DEFAULT_MODEL,
    DEVICE,
)


class ModelLoader:

    def load(self, model_name=DEFAULT_MODEL):

        print("=" * 60)
        print(f"Loading Model : {model_name}")
        print("=" * 60)

        model_id = MODELS[model_name]

        dtype = torch.float16 if DEVICE == "cuda" else torch.float32

        # --------------------------------------------------
        # Stable Diffusion 1.5
        # --------------------------------------------------

        if model_name == "Stable Diffusion 1.5":

            pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=dtype,
            )

        # --------------------------------------------------
        # SDXL
        # --------------------------------------------------

        elif model_name == "SDXL":

            pipe = StableDiffusionXLPipeline.from_pretrained(
                model_id,
                torch_dtype=dtype,
            )

        # --------------------------------------------------
        # FLUX
        # --------------------------------------------------

        elif model_name == "FLUX.1 Dev":

            raise NotImplementedError(
                "FLUX support will be added in the next phase."
            )

        else:

            raise ValueError(f"Unknown model : {model_name}")

        pipe = pipe.to(DEVICE)

        if DEVICE == "cuda":
            pipe.enable_attention_slicing()

        print("✅ Model Loaded Successfully.")

        return pipe