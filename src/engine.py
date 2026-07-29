from models.model_loader import ModelLoader
from src.generator import ImageGenerator


class TextToImageEngine:

    def __init__(self):

        self.current_model = None
        self.pipeline = None
        self.generator = None

    # ==========================================================
    # LOAD MODEL
    # ==========================================================

    def load_model(self, model_name):

        # Don't reload the same model
        if self.current_model == model_name:
            return

        print("=" * 60)
        print(f"Initializing {model_name}")
        print("=" * 60)

        loader = ModelLoader()

        self.pipeline = loader.load(model_name)

        self.generator = ImageGenerator(self.pipeline)

        self.current_model = model_name

        print(f"\n✅ {model_name} Ready!")

    # ==========================================================
    # GENERATE
    # ==========================================================

    def generate(
        self,
        model_name,
        prompt,
        negative_prompt="",
        width=512,
        height=512,
        steps=25,
        cfg=7.5,
        seed="",
        num_images=1,
    ):

        # Automatically load the selected model
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