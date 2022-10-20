import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import FreiHandDataset_Estimated as FreiHandDataset


def get_dataloader_train(data_path):
    joints_anno_file = "evaluation_xyz.json"
    camera_Ks_file = "evaluation_K.json"
    data_split_file = "FreiHand_split_ids.json"
    vertices_anno_file = "evaluation_verts.json"

    # Data Loaders
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.8705], std=[0.3358]),
        ]
    )

    dataset_train = FreiHandDataset(
        data_path,
        joints_anno_file,
        camera_Ks_file,
        data_split_file,
        vertices_anno_file,
        split="train",
        transform=transform,
        augment=True,
    )
    # dataset_val = FreiHandDataset(
    #     data_path,
    #     joints_anno_file,
    #     camera_Ks_file,
    #     data_split_file,
    #     vertices_anno_file,
    #     split="val",
    #     transform=transform,
    #     augment=False,
    # )
    batch_size = 1
    dataloader_train = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True, pin_memory=True)
    return dataloader_train
    # print(image.shape)
    # print(focal_len)
    # print(image_ref.shape)


def main(dataloader_train):
    from src.model import HandSilhouetteNet3

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = HandSilhouetteNet3(device)
    for inputs, focal_lens, image_refs, labels, dist_maps, meshes in dataloader_train:
        # image, focal_len, image_ref, label, dist_map, mesh = dataset_train[0]
        print(inputs.shape)
        print(focal_lens.shape)
        outputs = model(inputs, focal_lens, image_refs)
        print(outputs.shape)
        break


if __name__ == "__main__":
    dataloader = get_dataloader_train("./data/freihand")

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    main(dataloader)

    # inputs = torch.rand(1, 1, 224, 224)
    # focal_lens = torch.tensor([[531.9495, 532.2600]])
