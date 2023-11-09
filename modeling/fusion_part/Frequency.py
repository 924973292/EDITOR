import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pywt
import math
import torch.nn.functional as F
from modeling.backbones.vit_pytorch import PatchEmbed_overlap
from pytorch_wavelets import DWTForward, DWTInverse


def display_image(image_path, mode=1):
    pre_fix = '/13559197192/wyh/UNIReID/data/RGBNT201/train_171/'
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


class FrequencyIndex(nn.Module):
    def __init__(self,keep):
        super().__init__()
        self.DWT = DWTForward(J=4, wave='haar', mode='zero').cuda()
        self.IDWT = DWTInverse(wave='haar', mode='zero').cuda()
        self.keep = keep
        self.window_size = 16

    def show(self, x,writer=None, epoch=None,img_path=None):
        x = x[:12]
        num_rows = 2  # Number of rows in the display grid
        num_cols = 6  # Number of columns in the display grid
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 6))
        if x[0].shape[0]==3:
            x = x.permute(0, 2, 3, 1)
        # 循环遍历12个张量并可视化它们
        for i in range(12):
            mask_2d = x[i].cpu().numpy().astype(np.float32)
            # # 最小和最大值
            # min_value = np.min(mask_2d)
            # max_value = np.max(mask_2d)
            #
            # # 将数据映射到 0-255 范围
            # mapped_data = ((mask_2d - min_value) / (max_value - min_value) * 255).astype(np.uint8)

            row = i // num_cols
            col = i % num_cols

            # Display the masked image
            axes[row, col].imshow(mask_2d, cmap='bwr')
            axes[row, col].axis('off')
        plt.tight_layout()
        plt.show()
        if writer is not None:
            writer.add_figure('FREQUENCY', fig, global_step=epoch)

    def forward(self, x, y, z, img_path,pattern='a', mode=None, writer=None, step=None):
        batch_size, height, width = x.size(0), x.size(2), x.size(3)
        Ylx, Yhx = self.DWT(x)
        Yly, Yhy = self.DWT(y)
        Ylz, Yhz = self.DWT(z)
        low = (Ylx + Yly + Ylz) / 3
        high = [(Yhx[0] + Yhy[0] + Yhz[0]) / 3, (Yhx[1] + Yhy[1] + Yhz[1]) / 3, (Yhx[2] + Yhy[2] + Yhz[2]) / 3,
                (Yhx[3] + Yhy[3] + Yhz[3]) / 3]
        if pattern == 'low':
            low_ = torch.mean(low,dim=1)
            if self.training:
                self.show(low_, writer=writer, epoch=step)
            count_tensor = low_

        else:
            Inverse = self.IDWT((low, high))
            # 为Inverse统计每个16*16窗口内大于0的个数
            Inverse = torch.mean(Inverse, dim=1)
            if not self.training:
                # #找到000258_cam1_0_00这张图像在img_path中的索引,然后可视化它
                # search_value = '000258_cam1_0_00'
                # # 使用 PyTorch 的功能来查找索引
                # matching_indices = [i for i, value in enumerate(img_path) if value.split('.')[0] == search_value]
                # # 如果matching_indices非空，就可视化，否则不可视化，matching_indices需要变成适合当tensor索引的格式
                # if len(matching_indices) != 0:
                #     self.show(Inverse[int(matching_indices[0]):],writer=writer, epoch=step)
                pass



            # 创建一个用于存储统计结果的空张量，初始值都为0
            count_tensor = torch.zeros((batch_size, height // self.window_size, width // self.window_size), dtype=torch.int).cuda()

            # 循环遍历每个图像
            for batch_idx in range(batch_size):
                image = Inverse[batch_idx]  # 获取当前图像
                # 使用 unfold 函数来获取窗口视图
                unfolded = F.unfold(image.unsqueeze(0).unsqueeze(0), self.window_size, stride=self.window_size)
                # 将大于0的元素变成二进制，然后求和以统计大于0的元素数量
                count = unfolded.gt(0).sum(1)
                count = count.view(height // self.window_size, width // self.window_size)
                count_tensor[batch_idx] = count
                #获取每个图像的最大值的索引
        _, topk_indices = torch.topk(count_tensor.flatten(1), int(self.keep), dim=1)
        topk_indices = torch.sort(topk_indices, dim=1).values
        selected_tokens_mask = torch.zeros((batch_size, (height // self.window_size)  * (width // self.window_size)), dtype=torch.bool).cuda()
        selected_tokens_mask.scatter_(1, topk_indices, 1)
        # if self.training:
            # 判断batch中是否有000186存在，若存在，得到对应的索引original_index
            # search_value = '000186'
            # # 使用 PyTorch 的功能来查找索引
            # matching_indices = [i for i, value in enumerate(img_path) if value.split('_')[0] == search_value]
            # # 如果matching_indices非空，就可视化，否则不可视化，matching_indices需要变成适合当tensor索引的格式
            # if len(matching_indices) != 0:
            #     self.show(Inverse[int(matching_indices[0]):],writer=writer, epoch=step)
            # self.show(Inverse, writer=writer, epoch=step)
        return selected_tokens_mask
