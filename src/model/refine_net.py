import torch
import torch.nn as nn

# PyTorch3D rendering components
# from pytorch3d.renderer import (
#     BlendParams,
#     MeshRasterizer,
#     MeshRenderer,
#     PerspectiveCameras,
#     RasterizationSettings,
#     SoftSilhouetteShader,
#     TexturesVertex,
# )

# PyTorch3D data structures
# from pytorch3d.structures import Meshes
# from torchvision import models

##################################################
## RefineNet
##################################################


class RefineNet(nn.Module):
    def __init__(self, num_vertices):
        super(RefineNet, self).__init__()

        self.num_vertices = num_vertices

        self.net = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(output_size=(7, 7)),
        )

        self.fc = nn.Linear(7 * 7 * 256, num_vertices * 3)

    def forward(self, x):
        x = self.net(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    refine_net = RefineNet(num_vertices=778)
    out = refine_net(torch.rand(2, 2, 224, 224))
    print(out.shape)
