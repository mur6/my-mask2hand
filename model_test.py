import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model import HandSilhouetteNet3


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = HandSilhouetteNet3(device)

    # # x = torch.rand(1, 512)
    # # print(x.shape, focal_lens.shape)
    # out = model(, focal_lens)
    inputs = torch.rand(1, 1, 224, 224)
    focal_lens = torch.tensor([[531.9495, 532.2600]])
    image_refs = torch.rand(1, 224, 224)
    outputs = model(inputs, focal_lens, image_refs)
    print(outputs.shape)


main()
