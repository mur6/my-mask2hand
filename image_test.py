import json
import os
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


def main():
    image_name = "data/input_images/datasets/training/images/image_000000.jpg"
    orig_image = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)  # binary image with value {0, 255}
    image = 255 - orig_image
    print(image.shape)
    if image.shape != (224, 224):
        image_ref = cv2.resize(image, (224, 224))
        image_ref = cv2.threshold(image_ref, 127, 1, cv2.THRESH_BINARY)[1]
    else:
        image_ref = cv2.threshold(image, 127, 1, cv2.THRESH_BINARY)[1]

    # Extract contour and compute distance transform
    contour = cv2.Laplacian(image_ref, -1)
    contour = cv2.threshold(contour, 0, 1, cv2.THRESH_BINARY_INV)[1]
    dist_map = cv2.distanceTransform(contour, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    # plt.figure(figsize=(10, 10))
    fig, axs = plt.subplots(2, 2)
    axs = axs.flatten()

    print(orig_image.shape)
    axs[0].imshow(orig_image)
    axs[1].set_title("image:inverted")
    axs[1].imshow(image)
    axs[2].set_title("image_ref")
    axs[2].imshow(image_ref)
    # label
    axs[3].set_title("dist_map")
    axs[3].imshow(dist_map)
    # plt.axis("off")
    plt.show()

    dist_map = torch.tensor(dist_map)
    image_ref = torch.tensor(image_ref, dtype=torch.int)

    # image = Image.fromarray(image)
    # if self.transform:
    #     image = self.transform(image)

    label = torch.tensor(self.joints[index])
    mesh = torch.tensor(self.vertices[index])

    # focal_len = torch.tensor(self.focal_lengths[index])


if __name__ == "__main__":
    main()
