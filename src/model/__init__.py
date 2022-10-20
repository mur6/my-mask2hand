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

# PyTorch3D data structures
from pytorch3d.structures import Meshes
from torchvision import models

from .refine_net import RefineNet
from .shape_encoder import Encoder_with_Shape


class HandSilhouetteNet3(nn.Module):
    def __init__(self, device):
        super(HandSilhouetteNet3, self).__init__()
        # self.num_pca_comps = num_pca_comps
        self.device = device

        # Encoder (from shadow to global orient & pose PCAs)
        self.encoder = Encoder_with_Shape()  # (num_pca_comps=num_pca_comps)

        # RefineNet
        self.refine_net = RefineNet(num_vertices=778)

        # Configurations for rendering silhouettes

        # To blend the 100 faces we set a few parameters which control the opacity (gamma) and the sharpness (sigma) of edges.
        # 100 個の面をブレンドするために、エッジの不透明度 (ガンマ) とシャープネス (シグマ) を制御するいくつかのパラメーターを設定します。
        # The sigma value determines the sharpness of the peak in the normal distribution used in the blending function.
        # シグマ値は、ブレンド関数で使用される正規分布のピークの鋭さを決定します。
        # When we set sigma to be smaller, the silhouette will become shaper.
        # シグマを小さく設定すると、シルエットがよりシャープになります。
        self.blend_params = BlendParams(sigma=1e-4, gamma=1e-4, background_color=(1.0, 1.0, 1.0))
        # Define the settings for rasterization and shading. Here we set the output image to be of size 256x256.
        # ラスタライズとシェーディングの設定を定義します。ここでは、出力画像のサイズを 256x256 に設定します。
        # To form the blended image we use 100 faces for each pixel.
        # ブレンドされた画像を形成するために、各ピクセルに 100 個の面を使用します。
        # We also set bin_size and max_faces_per_bin to None which ensure that the faster coarse-to-fine rasterization method is used.
        # また、bin_size と max_faces_per_bin を None に設定して、より高速な粗から細へのラスター化方法が使用されるようにします。
        # Refer to rasterize_meshes.py for explanations of these parameters.
        # これらのパラメーターの説明については、rasterize_mehes.py を参照してください。
        # Refer to docs/notes/renderer.md for an explanation of the difference between naive and coarse-to-fine rasterization.
        # 単純なラスター化と粗いから細かいラスター化の違いの説明については、docs/notes/renderer.md を参照してください。
        self.raster_settings = RasterizationSettings(
            image_size=224,
            blur_radius=0.0,
            faces_per_pixel=100,
        )

    def forward(self, img, focal_lens, mask_gt):
        # Initialize a perspective camera
        # fx = fx_screen * 2.0 / image_width
        # fy = fy_screen * 2.0 / image_height
        # px = - (px_screen - image_width / 2.0) * 2.0 / image_width
        # py = - (py_screen - image_height / 2.0) * 2.0 / image_height
        self.cameras = PerspectiveCameras(device=self.device)

        # Create a silhouette mesh renderer by composing a rasterizer and a shader
        self.silhouette_renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=self.cameras, raster_settings=self.raster_settings),
            shader=SoftSilhouetteShader(blend_params=self.blend_params),
        )

        # Silhouette to pose PCAs & 3D global orientation & 3D translation & shape parameters
        code = self.encoder(img, focal_lens)
        print(f"code: {code.shape}")
        global_orient, transl = code[:, 0:3], code[:, 3:6]

        batch_size = code.shape[0]

        # Global orient & pose PCAs to 3D hand joints & reconstructed silhouette
        # rh_output = self.rh_model(
        #     betas=betas,
        #     global_orient=global_orient,
        #     hand_pose=pose_pcas,
        #     transl=transl,
        #     return_verts=True,
        #     return_tips=True,
        # )

        # Initialize each vertex to be white in color
        verts_rgb = torch.ones_like(rh_output.vertices)  # (B, V, 3)
        textures = TexturesVertex(verts_features=verts_rgb.to(self.device))

        # Coordinate transformation from FreiHand to PyTorch3D for rendering
        # [FreiHand] +X: right, +Y: down, +Z: in
        # [PyTorch3D] +X: left, +Y: up, +Z: in
        coordinate_transform = torch.tensor([[-1, -1, 1]]).to(self.device)

        mesh_faces = torch.tensor(self.rh_model.faces.astype(int)).to(self.device)

        # Create a Meshes object
        hand_meshes = Meshes(
            verts=[rh_output.vertices[i] * coordinate_transform for i in range(batch_size)],
            faces=[mesh_faces for i in range(batch_size)],
            textures=textures,
        )

        # Render the meshes
        silhouettes = self.silhouette_renderer(meshes_world=hand_meshes)
        silhouettes = silhouettes[..., 3]

        result = {
            "code": code,
            # "joints": output_joints,
            "silhouettes": silhouettes,
            "vertices": rh_output.vertices,
            # "refined_joints": refined_joints,
            # "refined_vertices": vertices,
            "betas": betas,
        }

        return result


if __name__ == "__main__":
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
