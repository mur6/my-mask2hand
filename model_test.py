import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from loss import criterion


def test_for_simple_model():
    from src.model import HandSilhouetteNet3

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


def visualize(silhouettes):
    print(silhouettes)
    fig, axs = plt.subplots(2, 2)
    axs = axs.flatten()
    axs[0].set_title("silhouettes")
    axs[0].imshow(silhouettes[0].detach().numpy())
    plt.show()


def test_for_full_simple_model():
    from model2 import HandSilhouetteNet3

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_path = "./models/MANO_RIGHT.pkl"
    num_pca_comps = 45
    model = HandSilhouetteNet3(model_path, num_pca_comps, device)

    # # x = torch.rand(1, 512)
    # # print(x.shape, focal_lens.shape)
    # out = model(, focal_lens)
    inputs = torch.rand(1, 3, 224, 224)
    focal_lens = torch.tensor([[531.9495, 532.2600]])
    image_refs = torch.rand(1, 224, 224)
    outputs = model(inputs, focal_lens, image_refs)
    silhouettes = outputs["silhouettes"]
    # loss = criterion(outputs, image_refs, labels, dist_maps, meshes, device)
    print(outputs.keys())
    visualize(silhouettes)


if __name__ == "__main__":
    test_for_full_simple_model()
