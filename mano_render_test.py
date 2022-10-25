import argparse
import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from pytorch3d.io import load_obj, load_objs_as_meshes
from pytorch3d.renderer import (
    BlendParams,
    DirectionalLights,
    FoVPerspectiveCameras,
    Materials,
    MeshRasterizer,
    MeshRenderer,
    PerspectiveCameras,
    PointLights,
    RasterizationSettings,
    SoftPhongShader,
    SoftSilhouetteShader,
    TexturesUV,
    TexturesVertex,
    look_at_view_transform,
)
from pytorch3d.structures import Meshes

# from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.vis.texture_vis import texturesuv_image_matplotlib
from torch.utils.data import DataLoader
from torchvision import transforms

from loss import FocalLoss, IoULoss


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


def visualize(image, silhouettes, mask):
    # print(silhouettes)
    fig, axs = plt.subplots(2, 2)
    axs = axs.flatten()
    axs[0].set_title("image")
    axs[0].imshow(image[0].permute(1, 2, 0).detach().numpy())
    axs[1].set_title("silhouettes")
    print(f"silhouettes:  {silhouettes.shape}")
    print(silhouettes)
    axs[1].imshow(silhouettes[0].detach().numpy())
    axs[2].set_title("silhouettes")
    axs[2].imshow(mask.detach().numpy())
    plt.show()


def load_image():
    image_name = "data/input_images/datasets/training/images/image_000000.jpg"
    mask_name = "data/input_images/datasets/training/masks/image_000000.png"
    mask = cv2.imread(mask_name, cv2.IMREAD_GRAYSCALE)
    mask = np.where(mask == 0, 0, 1)
    print(mask[0, 0])

    orig_image = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
    image = 255 - orig_image

    if image.shape != (224, 224):
        image_ref = cv2.resize(image, (224, 224))
        image_ref = cv2.threshold(image_ref, 127, 1, cv2.THRESH_BINARY)[1]
    else:
        image_ref = cv2.threshold(image, 127, 1, cv2.THRESH_BINARY)[1]

    # Extract contour and compute distance transform
    # contour = cv2.Laplacian(image_ref, -1)
    # contour = cv2.threshold(contour, 0, 1, cv2.THRESH_BINARY_INV)[1]
    # dist_map = cv2.distanceTransform(contour, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)

    image_ref = torch.tensor(image_ref, dtype=torch.int).unsqueeze(0)

    im = cv2.imread(image_name)
    im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im_rgb = torch.from_numpy(im_rgb.transpose(2, 0, 1)).clone()
    im_rgb = im_rgb.unsqueeze(0) / 255
    return im_rgb, image_ref, torch.from_numpy(mask)


def test_for_full_simple_model():
    from model2 import HandSilhouetteNet3

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_path = "./models/MANO_RIGHT.pkl"
    num_pca_comps = 45
    model = HandSilhouetteNet3(model_path, num_pca_comps, device)

    # image_name = "data/input_images/datasets/training/images/image_000000.jpg"
    inputs, image_refs, mask = load_image()
    mask_float = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
    print(f"mask: dtype={mask.dtype} shape={mask.shape}")

    print(inputs.dtype, image_refs.dtype)
    print(inputs.shape, image_refs.shape)
    # inputs = torch.rand(1, 3, 224, 224)
    focal_lens = torch.tensor([[8.0, 8.0]])
    # image_refs = torch.rand(1, 224, 224)
    # outputs = model(inputs, focal_lens, image_refs)

    # loss = criterion(outputs, image_refs, labels, dist_maps, meshes, device)

    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.75, momentum=0.9)
    criterion = IoULoss()
    # criterion = torchvision.ops.distance_box_iou_loss
    train_loss, val_loss = 0, 0
    # Train Phase
    model.train()
    for epoch in range(30):
        optimizer.zero_grad()
        outputs = model(inputs, focal_lens, image_refs)
        silhouettes = outputs["silhouettes"]
        # print(f"mask_float: dtype={mask_float.dtype} shape={mask_float.shape}")
        # print(f"silhouettes: dtype={silhouettes.dtype} shape={silhouettes.shape}")

        loss = criterion(silhouettes, mask_float)
        train_loss = loss.item()
        loss.backward()
        optimizer.step()
        # print(
        #     f"[Epoch {epoch}] Training Loss: {train_loss}, Validation Loss: {val_loss}, Last Learning Rate: {scheduler._last_lr}"
        # )
        print(f"[Epoch {epoch}] Training Loss: {train_loss}")
    model.eval()
    with torch.no_grad():
        outputs = model(inputs, focal_lens, image_refs)
        silhouettes = outputs["silhouettes"]
        # print(outputs.keys())
        visualize(inputs, silhouettes, mask)


import mano


def get_fov_cameras(device):
    R, T = look_at_view_transform(2.7, 0, 180)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
    return cameras


def get_perspective_cameras(device):
    focal_lens = torch.tensor([[0.1, 0.1]])
    cameras = PerspectiveCameras(focal_length=focal_lens * 2.0 / 224, device=device)
    return cameras


def get_renderer(device, cameras):
    lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])
    raster_settings = RasterizationSettings(
        image_size=512,
        blur_radius=0.0,
        faces_per_pixel=1,
    )
    # Create a Phong renderer by composing a rasterizer and a shader. The textured Phong shader will
    # interpolate the texture uv coordinates for each vertex, sample from a texture image and
    # apply the Phong lighting model
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=SoftPhongShader(device=device, cameras=cameras, lights=lights),
    )
    return renderer


def get_silhouette_renderer(cameras):
    raster_settings = RasterizationSettings(
        image_size=512,
        blur_radius=0.0,
        faces_per_pixel=1,
    )
    blend_params = BlendParams(sigma=1e-4, gamma=1e-4, background_color=(1.0, 1.0, 1.0))
    silhouette_renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=SoftSilhouetteShader(blend_params=blend_params),
    )
    return silhouette_renderer


def vizualize(target_tensor):
    target_np = target_tensor.cpu().numpy()
    plt.figure(figsize=(10, 10))
    plt.imshow(target_np)
    # plt.imshow(silhouettes.cpu().numpy())
    plt.axis("off")
    plt.show()


def vis_mano_right_hand_model(rh_model, output, mode=1):
    h_meshes = rh_model.hand_meshes(output)
    j_meshes = rh_model.joint_meshes(output)
    h_meshes[0].show()
    if mode == 1:
        # visualize hand mesh only
        h_meshes[0].show()
    elif mode == 2:
        # visualize joints mesh only
        j_meshes[0].show()
    elif mode == 3:
        # visualize hand and joint meshes
        from mano.utils import Mesh

        hj_meshes = Mesh.concatenate_meshes([h_meshes[0], j_meshes[0]])
        hj_meshes.show()


def make_random_mano_model():
    mano_model_path = "./models/MANO_RIGHT.pkl"
    n_comps = 45
    batch_size = 1

    rh_model = mano.load(
        model_path=mano_model_path, is_right=True, num_pca_comps=n_comps, batch_size=batch_size, flat_hand_mean=False
    )

    betas = torch.rand(batch_size, 10) * 0.1
    pose = torch.rand(batch_size, n_comps) * 0.1
    global_orient = torch.rand(batch_size, 3)
    transl = torch.rand(batch_size, 3)

    output = rh_model(
        betas=betas, global_orient=global_orient, hand_pose=pose, transl=transl, return_verts=True, return_tips=True
    )
    return rh_model, output


def show_hand_silhouettes():
    rh_model, output = make_random_mano_model()
    coordinate_transform = torch.tensor([[-1, -1, 1]])
    mesh_faces = torch.tensor(rh_model.faces.astype(int))
    batch_size = 1
    hand_meshes = Meshes(
        verts=[output.vertices[i] * coordinate_transform for i in range(batch_size)],
        faces=[mesh_faces for i in range(batch_size)],
        # verts=[rh_output.vertices[0]],
        # faces=[mesh_faces],
        # textures=textures,
    )
    print(hand_meshes)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cameras = get_fov_cameras(device)
    mesh = hand_meshes
    silhouettes = get_silhouette_renderer(cameras)(meshes_world=mesh)
    print(silhouettes.shape)
    vizualize(target_tensor=silhouettes[0, ..., 3])


def main():
    rh_model, output = make_random_mano_model()
    vis_mano_right_hand_model(rh_model, output, mode=3)


if __name__ == "__main__":
    main()
