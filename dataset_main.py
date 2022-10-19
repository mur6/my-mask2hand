import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import FreiHandDataset


def make_dataset(data_path):
    joints_anno_file = "evaluation_xyz.json"
    camera_Ks_file = "evaluation_K.json"
    data_split_file = "FreiHand_split_ids.json"
    vertices_anno_file = "evaluation_verts.json"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
    dataset_val = FreiHandDataset(
        data_path,
        joints_anno_file,
        camera_Ks_file,
        data_split_file,
        vertices_anno_file,
        split="val",
        transform=transform,
        augment=False,
    )


make_dataset("./data/freihand")
