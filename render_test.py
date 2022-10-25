from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# Util function for loading meshes
from pytorch3d.io import load_obj, load_objs_as_meshes

# PyTorch3D rendering components
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

# Data structures and functions for rendering
from pytorch3d.structures import Meshes

# from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.vis.texture_vis import texturesuv_image_matplotlib

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

obj_filename = Path("data/3d/hand.obj")

# Load obj file
mesh = load_objs_as_meshes([obj_filename], device=device)
# plt.figure(figsize=(7, 7))
# texture_image = mesh.textures.maps_padded()
# plt.imshow(texture_image.squeeze().cpu().numpy())
# plt.axis("off")
# plt.show()

# plt.figure(figsize=(7, 7))
# texturesuv_image_matplotlib(mesh.textures, subsample=None)
# plt.axis("off")
# plt.show()


R, T = look_at_view_transform(2.7, 0, 180)
cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

# Define the settings for rasterization and shading. Here we set the output image to be of size
# 512x512. As we are rendering images for visualization purposes only we will set faces_per_pixel=1
# and blur_radius=0.0. We also set bin_size and max_faces_per_bin to None which ensure that
# the faster coarse-to-fine rasterization method is used. Refer to rasterize_meshes.py for
# explanations of these parameters. Refer to docs/notes/renderer.md for an explanation of
# the difference between naive and coarse-to-fine rasterization.
raster_settings = RasterizationSettings(
    image_size=512,
    blur_radius=0.0,
    faces_per_pixel=1,
)

# Place a point light in front of the object. As mentioned above, the front of the cow is facing the
# -z direction.
lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])

# Create a Phong renderer by composing a rasterizer and a shader. The textured Phong shader will
# interpolate the texture uv coordinates for each vertex, sample from a texture image and
# apply the Phong lighting model
renderer = MeshRenderer(
    rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
    shader=SoftPhongShader(device=device, cameras=cameras, lights=lights),
)

# raster_settings = RasterizationSettings(
#     image_size=224,
#     blur_radius=0.0,
#     faces_per_pixel=100,
# )
blend_params = BlendParams(sigma=1e-4, gamma=1e-4, background_color=(1.0, 1.0, 1.0))
silhouette_renderer = MeshRenderer(
    rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
    shader=SoftSilhouetteShader(blend_params=blend_params),
)

images = renderer(mesh)
silhouettes = silhouette_renderer(meshes_world=mesh)
print(f"silhouettes: {silhouettes.shape}")
silhouettes = silhouettes[0, ..., 3]
print(f"silhouettes: {silhouettes.shape}")
plt.figure(figsize=(10, 10))
plt.imshow(images[0, ..., :3].cpu().numpy())
# plt.imshow(silhouettes.cpu().numpy())
plt.axis("off")
plt.show()
