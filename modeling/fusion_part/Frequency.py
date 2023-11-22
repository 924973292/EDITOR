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
from torchvision import transforms
from modeling.fusion_part.scale_channel import ScaleChannel

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
    def __init__(self, keep):
        super().__init__()
        self.DWT = DWTForward(J=4, wave='haar', mode='zero').cuda()
        self.IDWT = DWTInverse(wave='haar', mode='zero').cuda()
        self.gassian = transforms.GaussianBlur(kernel_size=5, sigma=0.1).cuda()
        self.laplace = torch.FloatTensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]]).cuda()
        self.keep = keep
        self.window_size = 16
    def show(self, x, writer=None, epoch=None, img_path=None, mode=1):
        x = x[:12]
        num_rows = 2  # Number of rows in the display grid
        num_cols = 6  # Number of columns in the display grid
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 6))
        if x[0].shape[0] == 3:
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
            if mode == 1:
                writer.add_figure('FREQUENCY_After', fig, global_step=epoch)
            elif mode == 2:
                writer.add_figure('FREQUENCY_Before', fig, global_step=epoch)

    def mask_single_stage(self, Inverse,window_size=16,mode='high'):
        batch_size, height, width = Inverse.size(0), Inverse.size(-2), Inverse.size(-1)
        if mode=='high':
            Inverse = Inverse[:,:,0]+Inverse[:,:,1]+Inverse[:,:,2]
        Inverse = torch.mean(Inverse, dim=1)
        # 创建一个用于存储统计结果的空张量，初始值都为0
        count_tensor = torch.zeros((batch_size, height // window_size, width // window_size),
                                   dtype=torch.int).cuda()

        # 循环遍历每个图像
        for batch_idx in range(batch_size):
            image = Inverse[batch_idx]  # 获取当前图像
            # 使用 unfold 函数来获取窗口视图
            unfolded = F.unfold(image.unsqueeze(0).unsqueeze(0), window_size, stride=window_size)
            # 将大于0的元素变成二进制，然后求和以统计大于0的元素数量
            count = unfolded.gt(0).sum(1)
            count = count.view(height // window_size, width // window_size)
            count_tensor[batch_idx] = count
            # 获取每个图像的最大值的索引
        _, topk_indices = torch.topk(count_tensor.flatten(1), int(self.keep), dim=1)
        topk_indices = torch.sort(topk_indices, dim=1).values
        selected_tokens_mask = torch.zeros((batch_size, (height // window_size) * (width // window_size)),
                                           dtype=torch.bool).cuda()
        selected_tokens_mask.scatter_(1, topk_indices, 1)
        return selected_tokens_mask

    def forward(self, x, y, z, img_path, pattern='a', mode=None, writer=None, step=None):
        Ylx, Yhx = self.DWT(x)
        # self.show(torch.mean(Ylx,dim=1), writer=writer, epoch=step, img_path=img_path, mode=mode)
        # self.show(torch.mean(Yhx[0][:,:,0],dim=1), writer=writer, epoch=step, img_path=img_path, mode=mode)
        # self.show(torch.mean(Yhx[0][:,:,1],dim=1), writer=writer, epoch=step, img_path=img_path, mode=mode)
        # self.show(torch.mean(Yhx[0][:,:,2],dim=1), writer=writer, epoch=step, img_path=img_path, mode=mode)
        Yly, Yhy = self.DWT(y)
        # self.show(torch.mean(Yly,dim=1), writer=writer, epoch=step, img_path=img_path, mode=mode)
        # self.show(torch.mean(Yhy[0][:,:,0],dim=1), writer=writer, epoch=step, img_path=img_path, mode=mode)
        # self.show(torch.mean(Yhy[0][:,:,1],dim=1), writer=writer, epoch=step, img_path=img_path, mode=mode)
        # self.show(torch.mean(Yhy[0][:,:,2],dim=1), writer=writer, epoch=step, img_path=img_path, mode=mode)
        if z!=None:
            Ylz, Yhz = self.DWT(z)
            # self.show(torch.mean(Ylz,dim=1), writer=writer, epoch=step, img_path=img_path, mode=mode)
            # self.show(torch.mean(Yhz[0][:,:,0],dim=1), writer=writer, epoch=step, img_path=img_path, mode=mode)
            # self.show(torch.mean(Yhz[0][:,:,1],dim=1), writer=writer, epoch=step, img_path=img_path, mode=mode)
            # self.show(torch.mean(Yhz[0][:,:,2],dim=1), writer=writer, epoch=step, img_path=img_path, mode=mode)
            low = (Ylx+Yly+Ylz)/3
            high = []
            for i in range(len(Yhx)):
                high.append((Yhx[i]+Yhy[i]+Yhz[i])/3)
        else:
            low = (Ylx + Yly) / 2
            high = []
            for i in range(len(Yhx)):
                high.append((Yhx[i] + Yhy[i]) / 2)
        # self.show(torch.mean(low,dim=1), writer=writer, epoch=step, img_path=img_path, mode=mode)
        # self.show(torch.mean(high[0][:,:,0],dim=1), writer=writer, epoch=step, img_path=img_path, mode=mode)
        # self.show(torch.mean(high[0][:,:,1],dim=1), writer=writer, epoch=step, img_path=img_path, mode=mode)
        # self.show(torch.mean(high[0][:,:,2],dim=1), writer=writer, epoch=step, img_path=img_path, mode=mode)
        Inverse = self.IDWT((low, high))
        # self.show(torch.mean(Inverse,dim=1), writer=writer, epoch=step, img_path=img_path, mode=mode)
        selected_tokens_mask = self.mask_single_stage(Inverse=Inverse, window_size=16, mode='low')
        #统计selected_tokens_mask非0的个数
        # print(selected_tokens_mask.sum()//x.shape[0])
        return selected_tokens_mask
