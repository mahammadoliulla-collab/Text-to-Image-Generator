from models.model_loader import ModelLoader
from src.generator import ImageGenerator


class TextToImageEngine:

    def __init__(self):

        print("=" * 60)
        print("Initializing Text-to-Image Engine")
        print("=" * 60)

        loader = ModelLoader()

        self.pipeline = loader.load()

        self.generator = ImageGenerator(self.pipeline)

        print("\n✅ Engine Ready!")

    def generate(
        self,
        prompt,
        negative_prompt="",
        width=512,
        height=512,
        steps=25,
        cfg=7.5,
        seed="",
        num_images=1,
    ):

        return self.generator.generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            steps=steps,
            cfg=cfg,
            seed=seed,
            num_images=num_images,
        )