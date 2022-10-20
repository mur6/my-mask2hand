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

##################################################
## Multi-head Encoder
##################################################


class Encoder_with_Shape(nn.Module):
    def __init__(self):
        super(Encoder_with_Shape, self).__init__()
        self.feature_extractor = models.resnet18(pretrained=True)
        self.feature_extractor.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        fc_in_features = self.feature_extractor.fc.in_features
        print(f"fc_in_features: {fc_in_features}")
        self.feature_extractor.fc = nn.Sequential(nn.Linear(fc_in_features, fc_in_features), nn.ReLU())

        self.rotation_estimator = nn.Sequential(
            nn.Linear(fc_in_features, fc_in_features // 2),
            nn.ReLU(),
            nn.Linear(fc_in_features // 2, fc_in_features // 4),
            nn.ReLU(),
            nn.Linear(fc_in_features // 4, 3),  # 3D global orientation
        )

        self.translation_estimator = nn.Sequential(
            nn.Linear(fc_in_features + 2, fc_in_features // 2),
            nn.ReLU(),
            nn.Linear(fc_in_features // 2, fc_in_features // 4),
            nn.ReLU(),
            nn.Linear(fc_in_features // 4, 3),  # 3D translation
        )

        # self.hand_shape_estimator = nn.Sequential(
        #     nn.Linear(fc_in_features, fc_in_features // 2),
        #     nn.ReLU(),
        #     nn.Linear(fc_in_features // 2, 10),  # MANO shape parameters
        # )

    def forward(self, x, focal_lens):
        x = self.feature_extractor(x)
        print(x.shape)
        # hand_pca = self.hand_pca_estimator(x)
        global_orientation = self.rotation_estimator(x)
        translation = self.translation_estimator(torch.cat([x, focal_lens], -1))
        # hand_shape = self.hand_shape_estimator(x)
        output = torch.cat([global_orientation, translation], -1)
        return output


if __name__ == "__main__":
    model = Encoder_with_Shape()
    focal_lens = torch.tensor([[531.9495, 532.2600]])
    # x = torch.rand(1, 512)
    # print(x.shape, focal_lens.shape)
    out = model(torch.rand(1, 1, 224, 224), focal_lens)
    print(out.shape)
