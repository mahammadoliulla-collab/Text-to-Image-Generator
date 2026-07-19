from models.model_loader import ModelLoader
from src.generator import ImageGenerator

from utils.input_handler import get_user_inputs
from utils.history_viewer import show_history


print("=" * 60)
print("TEXT TO IMAGE GENERATOR")
print("=" * 60)

print("\n1. Generate Image")
print("2. View Generation History")

choice = input("\nEnter Choice : ")

if choice == "2":

    show_history()

else:

    loader = ModelLoader()

    pipe = loader.load()

    generator = ImageGenerator(pipe)

    (
        prompt,
        negative_prompt,
        width,
        height,
        steps,
        cfg,
        seed,
        num_images,
    ) = get_user_inputs()

    generator.generate(
        prompt,
        negative_prompt,
        width,
        height,
        steps,
        cfg,
        seed,
        num_images,
    )