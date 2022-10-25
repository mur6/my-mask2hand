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

import mano


def create_mesh_from_mano(device):
    mano_model_path = "./models/MANO_RIGHT.pkl"
    num_pca_comps = 45
    rh_model = mano.load(model_path=mano_model_path, is_rhand=True, num_pca_comps=num_pca_comps, flat_hand_mean=True)
    code = torch.rand(1, 61)
    pose_pcas, global_orient, transl, betas = code[:, :-16], code[:, -16:-13], code[:, -13:-10], code[:, -10:]
    # Global orient & pose PCAs to 3D hand joints & reconstructed silhouette
    rh_output = rh_model(
        betas=betas,
        global_orient=global_orient,
        hand_pose=pose_pcas,
        transl=transl,
        return_verts=True,
        return_tips=True,
    )

    # Initialize each vertex to be white in color
    verts_rgb = torch.ones_like(rh_output.vertices)  # (B, V, 3)
    textures = TexturesVertex(verts_features=verts_rgb.to(device))

    # Coordinate transformation from FreiHand to PyTorch3D for rendering
    # [FreiHand] +X: right, +Y: down, +Z: in
    # [PyTorch3D] +X: left, +Y: up, +Z: in
    coordinate_transform = torch.tensor([[-1, -1, 1]]).to(device)

    mesh_faces = torch.tensor(rh_model.faces.astype(int)).to(device)

    # Create a Meshes object
    batch_size = 1
    verts = [rh_output.vertices[i] * coordinate_transform for i in range(batch_size)]
    faces = [mesh_faces for i in range(batch_size)]
    print(f"orig verts: {verts[0].shape}")
    print(f"orig faces: {faces[0].shape}")
    print(f"rh_output.vertices: {rh_output.vertices.shape}")
    print(f"mesh_faces: {mesh_faces.shape}")
    hand_meshes = Meshes(
        # verts=[rh_output.vertices[i] * coordinate_transform for i in range(batch_size)],
        # faces=[mesh_faces for i in range(batch_size)],
        verts=[rh_output.vertices[0]],
        faces=[mesh_faces],
        textures=textures,
    )
    return hand_meshes


def load_mesh_from_file(device):
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
    return mesh


def get_fov_cameras(device):
    R, T = look_at_view_transform(2.7, 0, 180)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
    return cameras


def get_perspective_cameras(device):
    # R, T = look_at_view_transform(2.7, 0, 180)
    focal_lens = torch.tensor([[8.0, 8.0]])
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


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cameras = get_perspective_cameras(device)
    mesh = create_mesh_from_mano(device)
    images = get_renderer(device, cameras)(mesh)
    silhouettes = get_silhouette_renderer(cameras)(meshes_world=mesh)
    print(f"silhouettes: {silhouettes.shape}")
    silhouettes = silhouettes[0, ..., 3]
    print(f"silhouettes: {silhouettes.shape}")
    vizualize(images[0, ..., :3])


main()
