import torch
import torch.nn as nn

# PyTorch3D rendering components
from pytorch3d.renderer import (
    BlendParams,
    MeshRasterizer,
    MeshRenderer,
    PerspectiveCameras,
    RasterizationSettings,
    SoftSilhouetteShader,
    TexturesVertex,
)

# from mano.lbs import vertices2joints


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cameras = PerspectiveCameras(device=device)
print(cameras)
raster_settings = RasterizationSettings(
    image_size=224,
    blur_radius=0.0,
    faces_per_pixel=100,
)
blend_params = BlendParams(sigma=1e-4, gamma=1e-4, background_color=(1.0, 1.0, 1.0))
silhouette_renderer = MeshRenderer(
    rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
    shader=SoftSilhouetteShader(blend_params=blend_params),
)
print(silhouette_renderer)
