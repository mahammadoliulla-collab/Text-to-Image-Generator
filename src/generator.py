import torch
import random
from datetime import datetime

from src.prompt_processor import PromptProcessor

from configs.config import (
    OUTPUT_FOLDER,
    IMAGE_WIDTH,
    IMAGE_HEIGHT,
    NUM_INFERENCE_STEPS,
    GUIDANCE_SCALE,
    NEGATIVE_PROMPT,
)

from utils.helper import create_output_folder
from utils.metadata import save_metadata
from utils.history import save_history


class ImageGenerator:

    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.processor = PromptProcessor()

        create_output_folder(OUTPUT_FOLDER)

    def generate(
        self,
        prompt,
        negative_prompt="",
        width="",
        height="",
        steps="",
        cfg="",
        seed="",
        num_images="1",
    ):

        # Clean Prompt
        prompt = self.processor.clean(prompt)

        # Default Negative Prompt
        if negative_prompt.strip() == "":
            negative_prompt = NEGATIVE_PROMPT

        # Width
        try:
            width = int(width)
        except:
            width = IMAGE_WIDTH

        # Height
        try:
            height = int(height)
        except:
            height = IMAGE_HEIGHT

        # Steps
        try:
            steps = int(steps)
        except:
            steps = NUM_INFERENCE_STEPS

        # CFG
        try:
            cfg = float(cfg)
        except:
            cfg = GUIDANCE_SCALE

        # Number of Images
        try:
            num_images = int(num_images)
        except:
            num_images = 1

        print("\nGenerating Images...")
        print("=" * 55)

        for i in range(num_images):

            # Seed
            if seed == "":
                current_seed = random.randint(0, 2**32 - 1)
            else:
                current_seed = int(seed)

            # Generator on the correct device
            generator = torch.Generator(
                device=self.pipeline.device
            ).manual_seed(current_seed)

            print(f"\nGenerating Image {i+1}/{num_images}")
            print(f"Seed : {current_seed}")

            image = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_inference_steps=steps,
                guidance_scale=cfg,
                generator=generator,
            ).images[0]

            # Timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            filename = f"generated_{timestamp}_{i+1}.png"

            save_path = f"{OUTPUT_FOLDER}/{filename}"

            # Save Image
            image.save(save_path)

            # Save Metadata
            save_metadata(
                image_path=save_path,
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                steps=steps,
                cfg=cfg,
                seed=current_seed,
            )

            # Save History
            save_history(
                {
                    "image": filename,
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "width": width,
                    "height": height,
                    "steps": steps,
                    "cfg": cfg,
                    "seed": current_seed,
                    "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
            )

            print(f"✅ Image Saved     : {save_path}")
            print(f"📝 Metadata Saved : {save_path.replace('.png', '.txt')}")
            print("📚 History Updated")

            # Display image (Colab) or open locally
            try:
                from IPython.display import display
                display(image)
            except Exception:
                image.show()

        print("\n" + "=" * 55)
        print("🎉 ALL IMAGES GENERATED SUCCESSFULLY")
        print("=" * 55)