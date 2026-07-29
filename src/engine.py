from models.model_loader import ModelLoader
from src.generator import ImageGenerator

from configs.config import DEFAULT_MODEL


class TextToImageEngine:

    def __init__(self):

        print("=" * 60)
        print("Initializing Text-to-Image Engine")
        print("=" * 60)

        self.loader = ModelLoader()

        self.current_model = None
        self.pipeline = None
        self.generator = None

        self.load_model(DEFAULT_MODEL)

        print("\n✅ Engine Ready!")

    # ======================================================
    # LOAD MODEL
    # ======================================================

    def load_model(self, model_name):

        if self.current_model == model_name:
            return

        print(f"\n🔄 Switching to: {model_name}")

        self.pipeline = self.loader.load(model_name)

        self.generator = ImageGenerator(self.pipeline)

        self.current_model = model_name

    # ======================================================
    # GENERATE
    # ======================================================

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
        model_name=DEFAULT_MODEL,
    ):

        self.load_model(model_name)

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