import random
from pathlib import Path

import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Util function for loading meshes
from pytorch3d.io import load_obj, load_objs_as_meshes

# PyTorch3D rendering components
from pytorch3d.renderer import (
    BlendParams,
    DirectionalLights,
    FoVPerspectiveCameras,
    HardPhongShader,
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
    look_at_rotation,
    look_at_view_transform,
)

# Data structures and functions for rendering
from pytorch3d.structures import Meshes

# from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.vis.texture_vis import texturesuv_image_matplotlib
from skimage import img_as_ubyte
from tqdm import tqdm

import mano

# import torchvision.transforms.functional as TF


device = torch.device("cpu")


def load_mesh_from_file(obj_filename):
    # Load the obj and ignore the textures and materials.
    verts, faces_idx, _ = load_obj(obj_filename)
    faces = faces_idx.verts_idx

    # Initialize each vertex to be white in color.
    verts_rgb = torch.ones_like(verts)[None]  # (1, V, 3)
    textures = TexturesVertex(verts_features=verts_rgb.to(device))
    # Create a Meshes object for the teapot. Here we have only one mesh in the batch.
    teapot_mesh = Meshes(verts=[verts.to(device)], faces=[faces.to(device)], textures=textures)
    return teapot_mesh


def main1(image_size=224):
    # Initialize a perspective camera.
    cameras = FoVPerspectiveCameras(device=device)

    # To blend the 100 faces we set a few parameters which control the opacity and the sharpness of
    # edges. Refer to blending.py for more details.
    blend_params = BlendParams(sigma=1e-4, gamma=1e-4)

    # Define the settings for rasterization and shading. Here we set the output image to be of size
    # 256x256. To form the blended image we use 100 faces for each pixel. We also set bin_size and max_faces_per_bin to None which ensure that
    # the faster coarse-to-fine rasterization method is used. Refer to rasterize_meshes.py for
    # explanations of these parameters. Refer to docs/notes/renderer.md for an explanation of
    # the difference between naive and coarse-to-fine rasterization.
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=np.log(1.0 / 1e-4 - 1.0) * blend_params.sigma,
        faces_per_pixel=100,
    )

    # Create a silhouette mesh renderer by composing a rasterizer and a shader.
    silhouette_renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=SoftSilhouetteShader(blend_params=blend_params),
    )

    # We will also create a Phong renderer. This is simpler and only needs to render one face per pixel.
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=0.0,
        faces_per_pixel=1,
    )
    # We can add a point light in front of the object.
    lights = PointLights(device=device, location=((2.0, 2.0, -2.0),))
    phong_renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=HardPhongShader(device=device, cameras=cameras, lights=lights),
    )
    return silhouette_renderer, phong_renderer


class Model(nn.Module):
    def __init__(self, meshes, renderer, image_ref, mask):
        super().__init__()
        self.meshes = meshes
        self.device = meshes.device
        self.renderer = renderer

        # Get the silhouette of the reference RGB image by finding all non-white pixel values.
        image_ref = torch.from_numpy((image_ref[..., :3].max(-1) != 1).astype(np.float32))
        image_ref = mask
        self.register_buffer("image_ref", image_ref)

        # Create an optimizable parameter for the x, y, z position of the camera.
        self.camera_position = nn.Parameter(
            # 3.0, 6.9, +2.5
            # -0.21, -0.5, +0.19
            # -0.15, -0.30, 0.39
            # -0.03, -0.30, 0.39
            # 0.0, 0.5, 0.5
            # 0.2590, -0.3444, 0.4757
            # 0.77, -0.4, 0.98
            torch.from_numpy(np.array([0.5, 1.0, 0.1], dtype=np.float32)).to(meshes.device)
        )
        image_size = 224
        xs = torch.linspace(2.0, -0.75, steps=image_size).unsqueeze(1)
        xs = xs.repeat(1, image_size)
        self.xs = torch.clamp(xs, min=0.0, max=1.0)

    def forward(self):

        # Render the image using the updated camera position. Based on the new position of the
        # camera we calculate the rotation and translation matrices
        R = look_at_rotation(self.camera_position[None, :], device=self.device)  # (1, 3, 3)
        T = -torch.bmm(R.transpose(1, 2), self.camera_position[None, :, None])[:, :, 0]  # (1, 3)

        image = self.renderer(meshes_world=self.meshes.clone(), R=R, T=T)

        # Calculate the silhouette loss
        loss = torch.sum((image[..., 3] - self.image_ref) ** 2)
        return loss, image


def main2(model, phong_renderer):
    # We will save images periodically and compose them into a GIF.
    filename_output = "./teapot_optimization_demo_02.gif"
    writer = imageio.get_writer(filename_output, mode="I", duration=0.3)

    # Create an optimizer. Here we are using Adam and we pass in the parameters of the model
    optimizer = torch.optim.Adam(model.parameters(), lr=0.025)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9)

    plt.figure(figsize=(10, 10))

    _, image_init = model()
    plt.subplot(1, 2, 1)
    plt.imshow(image_init.detach().squeeze().cpu().numpy()[..., 3])
    plt.grid(False)
    plt.title("Starting position")

    plt.subplot(1, 2, 2)
    plt.imshow(model.image_ref.cpu().numpy().squeeze())
    plt.grid(False)
    plt.title("Reference silhouette")
    plt.show()

    loop = tqdm(range(100))
    for i in loop:
        optimizer.zero_grad()
        loss, _ = model()
        loss.backward()
        optimizer.step()
        print(f"camera: {model.camera_position}")

        loop.set_description("Optimizing (loss %.4f)" % loss.data)

        if loss.item() < 200:
            break

        # Save outputs to create a GIF.
        if i % 5 == 0:
            R = look_at_rotation(model.camera_position[None, :], device=model.device)
            T = -torch.bmm(R.transpose(1, 2), model.camera_position[None, :, None])[:, :, 0]  # (1, 3)
            image = phong_renderer(meshes_world=model.meshes.clone(), R=R, T=T)
            image = image[0, ..., :3].detach().squeeze().cpu().numpy()
            image = img_as_ubyte(image)
            writer.append_data(image)

            # plt.figure()
            # plt.imshow(image[..., :3])
            # plt.title("iter: %d, loss: %0.2f" % (i, loss.data))
            # plt.axis("off")

    writer.close()


def viz1(silhouette, image_ref):
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(silhouette.squeeze()[..., 3])  # only plot the alpha channel of the RGBA image
    plt.grid(False)
    plt.subplot(1, 2, 2)
    plt.imshow(image_ref.squeeze())
    plt.grid(False)
    plt.show()


def load_image():
    image_name = "data/input_images/datasets/training/images/image_000032.jpg"
    mask_name = "data/input_images/datasets/training/masks/image_000032.png"
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


def main():
    seed = 42
    # Python random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    obj_filename = Path("data/3d/static_half_hand.obj")
    teapot_mesh = load_mesh_from_file(obj_filename)
    # obj_filename = Path("data/teapot.obj")
    silhouette_renderer, phong_renderer = main1()
    # Render the teapot providing the values of R and T.
    # Select the viewpoint using spherical angles
    distance = 0.55  # distance from camera to the object
    elevation = 120.0  # angle of elevation in degrees
    azimuth = 120.0  # No rotation so the camera is positioned on the +Z axis.

    # Get the position of the camera based on the spherical angles
    R, T = look_at_view_transform(distance, elevation, azimuth, device=device)

    silhouette = silhouette_renderer(meshes_world=teapot_mesh, R=R, T=T)
    image_ref = phong_renderer(meshes_world=teapot_mesh, R=R, T=T)
    silhouette = silhouette.cpu().numpy()
    image_ref = image_ref.cpu().numpy()
    image_ref2 = (image_ref[..., :3].max(-1) != 1).astype(np.float32)
    _, _, mask = load_image()
    mask = mask.unsqueeze(0)
    # mask2 = (mask[..., :3].max(-1) != 1).astype(np.float32)
    print(f"image_ref: {image_ref.shape}  image_ref2: {image_ref2.shape}  mask: {mask.shape}")
    # viz1(silhouette, image_ref)
    # viz1(silhouette, mask)
    # Initialize a model using the renderer, mesh and reference image
    image_size = 224
    xs = torch.linspace(2.0, -0.75, steps=image_size).unsqueeze(1)
    xs = xs.repeat(1, image_size)
    xs = torch.clamp(xs, min=0.0, max=1.0)
    model = Model(meshes=teapot_mesh, renderer=silhouette_renderer, image_ref=image_ref, mask=mask * xs).to(device)
    main2(model, phong_renderer)


main()
