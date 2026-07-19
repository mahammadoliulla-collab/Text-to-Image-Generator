def get_user_inputs():

    print("\n" + "=" * 50)
    print(" AI TEXT TO IMAGE GENERATOR ")
    print("=" * 50)

    prompt = input("Enter Prompt : ")

    negative_prompt = input(
        "Negative Prompt (Leave blank for default): "
    )

    width = input(
        "Width (Default 768): "
    )

    height = input(
        "Height (Default 768): "
    )

    steps = input(
        "Inference Steps (Default 35): "
    )

    cfg = input(
        "CFG Scale (Default 8.5): "
    )

    seed = input(
        "Seed (Leave blank for random): "
    )

    num_images = input(
        "Number of Images (Default 1): "
    )

    return (
        prompt,
        negative_prompt,
        width,
        height,
        steps,
        cfg,
        seed,
        num_images,
    )