import argparse
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
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


#     optimizer = optim.Adam(model.parameters(), lr = args.init_lr)
#     scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min', factor = 0.5, patience = 5, min_lr = 1e-6)

#     # Train model
#     train_loss_list = []
#     val_loss_list = []
#     last_lr_list = []

#     if args.resume:
#         checkpoint_file = 'model.pth'
#         checkpoint = torch.load(os.path.join(args.checkpoint_path, checkpoint_file))
#         model.load_state_dict(checkpoint['model_state_dict'])
#         optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#         scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
#         start_epoch = checkpoint['epoch'] + 1
#         print('Start Epoch: {}\n'.format(start_epoch))

#         df_loss = pd.read_csv(os.path.join(args.checkpoint_path, 'loss.csv'))
#         train_loss_list = df_loss['train_loss'].tolist()
#         val_loss_list = df_loss['val_loss'].tolist()
#         last_lr_list = df_loss['last_lr'].tolist()

#     train_model(model, dataloader_train, dataloader_val, criterion, optimizer, device, args.num_epochs, start_epoch, scheduler, train_loss_list, val_loss_list, last_lr_list, args.checkpoint_path)


# if __name__ == '__main__':
#     # Set the seed for reproducibility
#     torch.manual_seed(0)
#     np.random.seed(0)
#     random.seed(0)

#     # args
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--data_path', type = str, default = './dataset/freihand')
#     parser.add_argument('--checkpoint_path', type = str, default = './checkpoint')
#     parser.add_argument('--batch_size', type = int, default = 32)
#     parser.add_argument('--num_epochs', type = int, default = 150)
#     parser.add_argument('--init_lr', type = float, default = 1e-4)
#     parser.add_argument('--num_pcs', type = int, default = 45, help = 'number of pose PCs (ex: 6, 45)')
#     parser.add_argument('--resume', action = 'store_true')
#     args = parser.parse_args()

#     main(args)


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
    focal_lens = torch.tensor([[531.9495, 532.2600]])
    # image_refs = torch.rand(1, 224, 224)
    # outputs = model(inputs, focal_lens, image_refs)

    # loss = criterion(outputs, image_refs, labels, dist_maps, meshes, device)

    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0, momentum=0.9)
    criterion = IoULoss()
    # criterion = torchvision.ops.distance_box_iou_loss
    train_loss, val_loss = 0, 0
    # Train Phase
    model.train()
    for epoch in range(5):
        optimizer.zero_grad()
        outputs = model(inputs, focal_lens, image_refs)
        silhouettes = outputs["silhouettes"]
        # print(f"mask_float: dtype={mask_float.dtype} shape={mask_float.shape}")
        # print(f"silhouettes: dtype={silhouettes.dtype} shape={silhouettes.shape}")

        loss = criterion(silhouettes, mask_float)
        print(loss)
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


if __name__ == "__main__":
    test_for_full_simple_model()
