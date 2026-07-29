from models.img2img_loader import Img2ImgLoader
from src.img2img_generator import Img2ImgGenerator


class Img2ImgEngine:

    def __init__(self):

        print("=" * 60)
        print("Initializing Image-to-Image Engine")
        print("=" * 60)

        loader = Img2ImgLoader()

        self.pipeline = loader.load()

        self.generator = Img2ImgGenerator(self.pipeline)

        print("✅ Img2Img Engine Ready!")

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

        return self.generator.generate(
            image=image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            strength=strength,
            steps=steps,
            cfg=cfg,
            seed=seed,
        )