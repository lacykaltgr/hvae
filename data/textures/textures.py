import subprocess
import os
import pickle

from src.elements.dataset import _DataSet
import os

class TexturesDataset(_DataSet):
    def __init__(self, type="natural", image_size=20, whitening="new"):
        self.type = type
        assert self.type in ["natural", "texture"]
        self.image_size = image_size
        self.whitening = whitening
        super(TexturesDataset, self).__init__()

    def load(self):
        if not len(os.listdir('data/textures/datasets')) == 4:
            run_download_script()

        if self.type == "natural":
            train, val, test = load_natural_ds(image_size=self.image_size)
        elif self.type == "texture":
            train, val, test = load_texture_ds(image_size=self.image_size, whitening=self.whitening)
        else:
            raise ValueError("Invalid type")

        return train, val, test


"""
DOWNLOAD
"""

def run_download_script():
    os.makedirs('data/textures/datasets/', exist_ok=True)
    script_path = 'data/textures/download_datasets.sh'
    process = subprocess.run(['bash', script_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if process.returncode == 0:
        print("Shell script completed successfully.")
    else:
        print("Shell script encountered an error.")


"""
LOAD
"""
def load_natural_ds(image_size=20):
    if image_size == 20:
        path = "data/textures/datasets/fakelabeled_natural_commonfiltered_640000_20px.pkl"
    elif image_size == 40:
        path = "data/textures/datasets/fakelabeled_natural_commonfiltered_640000_40px.pkl"
    elif image_size == 50:
        path = "data/textures/datasets/fakelabeled_natural_commonfiltered_640000_50px.pkl"
    else:
        raise ValueError('Image size can be either 20px, 40px or 50px')

    with open(path, 'rb') as dataset_file:
        data = pickle.load(dataset_file)

    train_val_split = int(0.8 * len(data["train_images"]))

    train_images = data["train_images"][: train_val_split]
    #train_labels = data["train_labels"][: train_val_split]
    val_images = data["train_images"][train_val_split:]
    #val_labels = data["train_labels"][train_val_split:]
    test_images = data["test_images"]
    #test_labels = data["test_labels"]
    return train_images, val_images, test_images


def load_texture_ds(image_size=20, whitening="new"):
    if image_size == 20:
        path = "datasets/labeled_texture_oatleathersoilcarpetbubbles_commonfiltered_640000_20px.pkl"
    elif image_size == 40:
        if whitening == "old":
            path = "data/textures/datasets/labeled_texture_oatleathersoilcarpetbubbles_commonfiltered_640000_40px.pkl"
        elif whitening == "new":
            path = "data/textures/datasets/labeled_texture_oatleathersoilcarpetbubbles_commonfiltered_640000_40px.pkl"#"/datasets/labeled_texture_oatleathersoilcarpetbubbles_commonfiltered_naturalPCA_640000_40px.pkl"
        else:
            raise TypeError
    elif image_size == 50:
        path = "data/textures/datasets/labeled_texture_oatleathersoilcarpetbubbles_commonfiltered_640000_50px.pkl"
    else:
        raise ValueError('Image size can be either 20px, 40px or 50px')

    with open(path, 'rb') as dataset_file:
        data = pickle.load(dataset_file)

    train_val_split = int(0.8 * len(data["train_images"]))
    train_images = data["train_images"][: train_val_split]
    train_labels = data["train_labels"][: train_val_split]
    val_images = data["train_images"][train_val_split:]
    val_labels = data["train_labels"][train_val_split:]
    test_images = data["test_images"]
    test_labels = data["test_labels"]

    return (train_images, train_labels), (val_images, val_labels), (test_images, test_labels)





