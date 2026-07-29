import random
import time
from datetime import datetime

import torch
from PIL import Image

from configs.config import (
    OUTPUT_FOLDER,
    NEGATIVE_PROMPT,
)

from utils.helper import create_output_folder
from utils.metadata import save_metadata
from utils.history import save_history


class Img2ImgGenerator:

    def __init__(self, pipeline):

        self.pipeline = pipeline

        create_output_folder(OUTPUT_FOLDER)

    def generate(
        self,
        image,
        prompt,
        negative_prompt="",
        strength=0.75,
        steps=25,
        cfg=7.5,
        seed="",
    ):

        if negative_prompt.strip() == "":
            negative_prompt = NEGATIVE_PROMPT

        if seed == "":
            current_seed = random.randint(0, 2**32 - 1)
        else:
            current_seed = int(seed)

        generator = torch.Generator(
            device=self.pipeline.device
        ).manual_seed(current_seed)

        image = image.convert("RGB")

        start = time.time()

        output = self.pipeline(
            prompt=prompt,
            image=image,
            negative_prompt=negative_prompt,
            strength=strength,
            guidance_scale=cfg,
            num_inference_steps=steps,
            generator=generator,
        ).images[0]

        generation_time = round(time.time() - start, 2)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        filename = f"img2img_{timestamp}.png"

        save_path = f"{OUTPUT_FOLDER}/{filename}"

        output.save(save_path)

        save_metadata(
            image_path=save_path,
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=output.width,
            height=output.height,
            steps=steps,
            cfg=cfg,
            seed=current_seed,
        )

        save_history(
            {
                "image": filename,
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "width": output.width,
                "height": output.height,
                "steps": steps,
                "cfg": cfg,
                "seed": current_seed,
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
        )

        return {
            "image_path": save_path,
            "generation_time": generation_time,
            "seed": current_seed,
            "width": output.width,
            "height": output.height,
        }