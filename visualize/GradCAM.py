from visualize.grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    deprocess_image, \
    preprocess_image
import cv2
from PIL import Image, ImageFile
import numpy as np
import os
from config import cfg
import argparse
from data import make_dataloader
from modeling import make_model
from utils.logger import setup_logger
import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UniSReID Testing")
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

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("UniSReID", output_dir, if_train=False)
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)


    def reshape_transform(tensor):
        if len(tensor) == 4:
            tensor = tensor[-1][:, 1:, :]
        else:
            tensor = tensor[1][:, 1:, :]
        result = tensor.reshape(tensor.size(0), 16, 8, tensor.size(2))
        result = result.transpose(2, 3).transpose(1, 2)
        return result


    class Newdict(dict):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def size(self, input):
            if input == -1 or input == 3:
                return 128
            elif input == -2 or input == 2:
                return 256
            elif input == -3 or input == 1:
                return 3
            else:
                return 1


    device = "cuda"
    # 1.加载模型
    model = make_model(cfg, num_class=num_classes, camera_num= camera_num, view_num=view_num)
    model.load_param("/15127306268/wyh/MM/RGBNT100/vit_top_re_bt/UniSReIDbest.pth")
    model.eval()
    # 2.选择目标层
    # target_layer = model.layer4
    model.to(device)
    target_layer = [model.reconstruct]
    for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(train_loader_normal):
        with torch.no_grad():
            img = Newdict({'RGB': img['RGB'].to(device),
                           'NI': img['NI'].to(device),
                           'TI': img['TI'].to(device)})
            input_tensor = img
        print(imgpath, pid)
        cam = GradCAM(model=model, target_layers=target_layer, reshape_transform=reshape_transform)

        grayscale_cam = cam(input_tensor=input_tensor, eigen_smooth=True)  # [batch, 224,224

        def show_cam(index, imgpath, grayscale_cam):
            index = int(index)
            imgpath = '/15127306268/wyh/MM/data/RGBNT100/rgbir/bounding_box_train/' + imgpath[index]
            grayscale_cam = grayscale_cam[index]
            if cfg.DATASETS.NAMES == 'RGBNT100':
                img = cv2.imread(imgpath, 1)
                rgb_image = img[:,:256,:]
            else:
                rgb_image = cv2.imread(imgpath, 1)
            rgb_image = cv2.resize(rgb_image, (128, 256))
            rgb_image = np.float32(rgb_image) / 255

            visualization = show_cam_on_image(rgb_image, grayscale_cam)  # (224, 224, 3)
            cv2.imwrite('{:}.jpg'.format(index), visualization)
        for i in range(cfg.TEST.IMS_PER_BATCH):
            show_cam(i, imgpath, grayscale_cam)
        break