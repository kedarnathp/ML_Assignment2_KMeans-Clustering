
from matplotlib import pyplot as plt
import numpy as np

def get_image(image_path):
    image = plt.imread(image_path)
    return image/255.0


def show_image(image):
    plt.imshow(image)
    plt.show()

def save_image(image, image_path):
    plt.imsave(image_path, image)


def error(original_image: np.ndarray, clustered_image: np.ndarray) -> float:
    mse = np.mean((original_image - clustered_image) ** 2)
    return mse
