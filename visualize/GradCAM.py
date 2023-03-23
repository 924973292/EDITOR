from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    deprocess_image, \
    preprocess_image
from torchvision.models import resnet50
import cv2
import numpy as np
import argparse
import os
from modeling.make_model import make_model
from config import cfg

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
parser = argparse.ArgumentParser(description="ReID Baseline Training")
parser.add_argument(
    "--config_file", default="", help="path to config file", type=str
)
parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                    nargs=argparse.REMAINDER)

args = parser.parse_args()

if args.config_file != "":
    cfg.merge_from_file(args.config_file)
cfg.merge_from_list(args.opts)
cfg.freeze()


def reshape_transform(tensor, height=21, weight=10, pattern='d'):
    if pattern == 'd':
        return tensor
    elif pattern == 't':
        tensor = tensor[:, 1:, :]
        result = tensor.reshape(tensor.size(0), height, weight, tensor.size(2))
        result = result.transpose(2, 3).transpose(1, 2)
    elif pattern == 'r':
        tensor = tensor[:, 1:, :]
        result = tensor.reshape(tensor.size(0), 16, 8, tensor.size(2))
        result = result.transpose(2, 3).transpose(1, 2)
    return result


# 1.加载模型
# model = make_model(cfg, num_class=751, camera_num=6)
# model.load_param("resnet50.pth")
# model = make_model(cfg, num_class=751, camera_num=6,view_num=0)
model = make_model(cfg, num_class=751, camera_num=6, view_num=0)
model.load_param("./resnet50_180.pth")
model.eval()
# 2.选择目标层
# target_layer = model.layer4
target_layer = [model.former_LPU]
image_path = './0319_c6s1_072151_00.jpg'
rgb_img = cv2.imread(image_path, 1)[:, :, ::-1]  # 1是读取rgb
# imread返回从指定路径加载的图像
rgb_img = cv2.imread(image_path, 1)  # imread()读取的是BGR格式
rgb_img = cv2.resize(rgb_img, (128, 256))
rgb_img = np.float32(rgb_img) / 255

# preprocess_image作用：归一化图像，并转成tensor
input_tensor = preprocess_image(rgb_img, mean=[0.5, 0.5, 0.5],
                                std=[0.5, 0.5, 0.5])  # torch.Size([1, 3, 224, 224])
# Create an input tensor image for your model..
# Note: input_tensor can be a batch tensor with several images!


# ----------------------------------------
'''
3)初始化CAM对象，包括模型，目标层以及是否使用cuda等
'''
# Construct the CAM object once, and then re-use it on many images:
cam = GradCAM(model=model, target_layers=target_layer, use_cuda=True, reshape_transform=reshape_transform)
'''
4)选定目标类别，如果不设置，则默认为分数最高的那一类
'''
# If target_category is None, the highest scoring category
# will be used for every image in the batch.
# target_category can also be an integer, or a list of different integers
# for every image in the batch.
# target_category = 0
# 指定类：target_category = 281

'''
5)计算ca
'''
# You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
grayscale_cam = cam(input_tensor=input_tensor,target_layer=target_layer)  # [batch, 224,224
# ---------------------------------
'''target_category
6)展示热力图并保存
'''
# In this example grayscale_cam has only one image in the batch:
# 7.展示热力图并保存, grayscale_cam是一个batch的结果，只能选择一张进行展示
grayscale_cam = grayscale_cam[0]
visualization = show_cam_on_image(rgb_img, grayscale_cam)  # (224, 224, 3)
cv2.imwrite(f'first_try.jpg', visualization)
