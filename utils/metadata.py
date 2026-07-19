from datetime import datetime


def save_metadata(
    image_path,
    prompt,
    negative_prompt,
    width,
    height,
    steps,
    cfg,
    seed,
):
    """
    Save generation settings alongside each image.
    """

    metadata_path = image_path.replace(".png", ".txt")

    with open(metadata_path, "w", encoding="utf-8") as file:

        file.write("=" * 50 + "\n")
        file.write("TEXT TO IMAGE GENERATOR METADATA\n")
        file.write("=" * 50 + "\n\n")

        file.write(f"Generated On : {datetime.now()}\n\n")

        file.write(f"Prompt :\n{prompt}\n\n")

        file.write(f"Negative Prompt :\n{negative_prompt}\n\n")

        file.write(f"Width : {width}\n")
        file.write(f"Height : {height}\n")
        file.write(f"Inference Steps : {steps}\n")
        file.write(f"CFG Scale : {cfg}\n")
        file.write(f"Seed : {seed}\n")