import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch.nn.functional as F
from pytorch_wavelets import DWTForward, DWTInverse


def display_image(image_path, mode=1):
    pre_fix = '/13994058190/WYH/EDITOR/data/RGBNT201/train_171/'
    if mode == 1:
        pre_fix = pre_fix + 'RGB/'
    elif mode == 2:
        pre_fix = pre_fix + 'NI/'
    elif mode == 3:
        pre_fix = pre_fix + 'TI/'
    image = Image.open(pre_fix + image_path)
    resized_image = image.resize((128, 256))  # Resize to 256x128
    plt.imshow(resized_image)
    plt.axis('off')
    plt.show()


# Visualize the mask on the image
def visualize_multiple_masks(images, masks, mode, pre_fix, writer=None, epoch=None):
    num_images_to_display = 12  # Number of images to display
    images = images[:num_images_to_display]
    num_rows = 2  # Number of rows in the display grid
    num_cols = 6  # Number of columns in the display grid
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 6))

    for i in range(num_images_to_display):
        if mode == 1:
            prefix = pre_fix + 'RGB/'
        elif mode == 2:
            prefix = pre_fix + 'NI/'
        elif mode == 3:
            prefix = pre_fix + 'TI/'
        # Load the original image
        image = Image.open(prefix + images[i])
        original_image = image.resize((128, 256))  # Resize to 256x128

        # Convert the image to numpy array
        original_np = np.array(original_image)

        row = i // num_cols
        col = i % num_cols

        # Display the masked image
        axes[row, col].imshow(original_np)
        axes[row, col].axis('off')
    plt.tight_layout()
    plt.show()
    if writer is not None:
        if mode == 1:
            sign = 'RGB'
        elif mode == 2:
            sign = 'NIR'
        elif mode == 3:
            sign = 'TIR'
        writer.add_figure('FREQUENCY_' + sign, fig, global_step=epoch)


class Frequency_based_Token_Selection(nn.Module):
    def __init__(self, keep,stride=16):
        super().__init__()
        self.DWT = DWTForward(J=4, wave='haar', mode='zero').cuda()
        self.IDWT = DWTInverse(wave='haar', mode='zero').cuda()
        self.keep = keep
        self.window_size = 16
        self.stride = stride
    def show(self, x, writer=None, epoch=None, img_path=None, mode=1):
        x = x[:12]
        num_rows = 2  # Number of rows in the display grid
        num_cols = 6  # Number of columns in the display grid
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 6))
        if x[0].shape[0] == 3:
            x = x.permute(0, 2, 3, 1)
        for i in range(12):
            mask_2d = x[i].cpu().numpy().astype(np.float32)
            row = i // num_cols
            col = i % num_cols
            axes[row, col].imshow(mask_2d, cmap='bwr')
            axes[row, col].axis('off')
        plt.tight_layout()
        plt.show()
        if writer is not None:
            if mode == 1:
                writer.add_figure('FREQUENCY_After', fig, global_step=epoch)
            elif mode == 2:
                writer.add_figure('FREQUENCY_Before', fig, global_step=epoch)

    def mask(self, Inverse,window_size=16):
        batch_size, height, width = Inverse.size(0), Inverse.size(-2), Inverse.size(-1)
        Inverse = torch.mean(Inverse, dim=1)
        # create a tensor to store the count of non-zero elements
        count_tensor = torch.zeros((batch_size, height //self.stride, width // self.stride),
                                   dtype=torch.int).cuda()
        # For each image in the batch
        for batch_idx in range(batch_size):
            image = Inverse[batch_idx]  # 获取当前图像
            # With a sliding window, unfold the image into a tensor
            unfolded = F.unfold(image.unsqueeze(0).unsqueeze(0), window_size, stride=self.stride)
            # Turns elements greater than 0 into binary, then sums to count the number of elements greater than 0
            count = unfolded.gt(0).sum(1)
            count = count.view(height // self.stride, width // self.stride)
            count_tensor[batch_idx] = count
            # Get the index of the maximum value of each image
        _, topk_indices = torch.topk(count_tensor.flatten(1), int(self.keep), dim=1)
        topk_indices = torch.sort(topk_indices, dim=1).values
        selected_tokens_mask = torch.zeros((batch_size, (height // self.stride) * (width // self.stride)),
                                           dtype=torch.bool).cuda()
        selected_tokens_mask.scatter_(1, topk_indices, 1)
        return selected_tokens_mask

    def forward(self, x, y, z, img_path, pattern='a', mode=None, writer=None, step=None):
        Ylx, Yhx = self.DWT(x)
        Yly, Yhy = self.DWT(y)
        if z!=None:
            Ylz, Yhz = self.DWT(z)
            low = (Ylx+Yly+Ylz)/3
            high = []
            for i in range(len(Yhx)):
                high.append((Yhx[i]+Yhy[i]+Yhz[i])/3)
        else:
            low = (Ylx + Yly) / 2
            high = []
            for i in range(len(Yhx)):
                high.append((Yhx[i] + Yhy[i]) / 2)

        Inverse = self.IDWT((low, high))
        selected_tokens_mask = self.mask(Inverse=Inverse, window_size=self.window_size)

        return selected_tokens_mask
