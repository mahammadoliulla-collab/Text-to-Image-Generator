import os

def create_output_folder(path):

    if not os.path.exists(path):
        os.makedirs(path)