import os
import zipfile

def create_output_folder(path):

    if not os.path.exists(path):
        os.makedirs(path)
def create_zip(image_paths):

    zip_path = "outputs/generated_images.zip"

    with zipfile.ZipFile(zip_path, "w") as zipf:

        for image in image_paths:

            zipf.write(
                image,
                arcname=os.path.basename(image)
            )

    return zip_path