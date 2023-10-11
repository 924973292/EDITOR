import numpy as np
from PIL import Image
import torch.utils.data as data


class SYSUData(data.Dataset):
    def __init__(self, data_dir, transform=None, colorIndex=None, thermalIndex=None):
        data_dir = '../Datasets/SYSU-MM01/'
        # Load training images (path) and labels
        train_color_image = np.load(data_dir + 'train_rgb_resized_img.npy')
        self.train_color_label = np.load(data_dir + 'train_rgb_resized_label.npy')

        train_thermal_image = np.load(data_dir + 'train_ir_resized_img.npy')
        self.train_thermal_label = np.load(data_dir + 'train_ir_resized_label.npy')

        # BGR to RGB
        self.train_color_image = train_color_image
        self.train_thermal_image = train_thermal_image
        self.transform = transform
        self.cIndex = colorIndex
        self.tIndex = thermalIndex

    def __getitem__(self, index):
        img1, target1 = self.train_color_image[self.cIndex[index]], self.train_color_label[self.cIndex[index]]
        img2, target2 = self.train_thermal_image[self.tIndex[index]], self.train_thermal_label[self.tIndex[index]]

        img1 = self.transform(img1)
        img2 = self.transform(img2)

        return img1, img2, target1, target2

    def __len__(self):
        return len(self.train_color_label)


class RegDBData(data.Dataset):
    def __init__(self, data_dir, transform=None, colorIndex=None, thermalIndex=None):
        # Load training images (path) and labels
        data_dir = '/15127306268/wyh/MM/data/RGBNT201-C/'
        train_color_list = data_dir + 'idx/train_R' + '.txt'
        train_thermal_list = data_dir + 'idx/train_N' + '.txt'

        color_img_file, train_color_label = load_data(train_color_list)
        thermal_img_file, train_thermal_label = load_data(train_thermal_list)

        train_color_image = []
        train_color_image_path = []
        train_color_image_camid = []
        train_color_image_viewid = []
        for i in range(len(color_img_file)):
            img = Image.open(data_dir + color_img_file[i])
            train_color_image_path.append(color_img_file[i])
            jpg_name = color_img_file[i].split('/')[-1]
            camid = int(jpg_name.split('_')[1][3])
            camid -= 1  # index starts from 0
            train_color_image_camid.append(camid)
            viewid = int(jpg_name.split('_')[2][0])
            train_color_image_viewid.append(viewid)
            img = img.resize((128, 256), Image.ANTIALIAS)
            pix_array = np.array(img)
            train_color_image.append(pix_array)
        train_color_image = np.array(train_color_image)
        train_color_image_path = np.array(train_color_image_path)
        train_color_image_camid = np.array(train_color_image_camid)
        train_color_image_viewid = np.array(train_color_image_viewid)

        train_thermal_image = []
        train_thermal_image_path = []
        train_thermal_image_camid = []
        train_thermal_image_viewid = []
        for i in range(len(thermal_img_file)):
            img = Image.open(data_dir + thermal_img_file[i])
            img = img.resize((128, 256), Image.ANTIALIAS)
            train_thermal_image_path.append(thermal_img_file[i])
            jpg_name = thermal_img_file[i].split('/')[-1]
            camid = int(jpg_name.split('_')[1][3])
            camid -= 1  # index starts from 0
            train_thermal_image_camid.append(camid)
            viewid = int(jpg_name.split('_')[2][0])
            train_thermal_image_viewid.append(viewid)
            pix_array = np.array(img)
            train_thermal_image.append(pix_array)
        train_thermal_image = np.array(train_thermal_image)

        # BGR to RGB
        self.train_color_image = train_color_image
        self.train_color_label = train_color_label
        self.train_color_image_path = train_color_image_path
        self.train_color_image_camid = train_color_image_camid
        self.train_color_image_viewid = train_color_image_viewid

        # BGR to RGB
        self.train_thermal_image = train_thermal_image
        self.train_thermal_label = train_thermal_label
        self.train_thermal_image_path = train_thermal_image_path
        self.train_thermal_image_camid = train_thermal_image_camid
        self.train_thermal_image_viewid = train_thermal_image_viewid

        self.transform = transform
        self.cIndex = colorIndex
        self.tIndex = thermalIndex

    def __getitem__(self, index):

        img1, target1 = self.train_color_image[self.cIndex[index]], self.train_color_label[self.cIndex[index]]
        img2, target2 = self.train_thermal_image[self.tIndex[index]], self.train_thermal_label[self.tIndex[index]]

        img1 = self.transform(img1)
        img2 = self.transform(img2)
        img = [img1, img2]
        pid = target1
        camid = self.train_color_image_camid[self.cIndex[index]]
        trackid = -1
        img_path = self.train_color_image_path[self.cIndex[index]]
        return img, pid, camid, trackid, img_path.split('/')[-1]

    def __len__(self):
        return len(self.train_color_label)


class TestData(data.Dataset):
    def __init__(self, test_img_file, test_label, transform=None, img_size=(128, 256)):
        test_image = []
        for i in range(len(test_img_file)):
            img = Image.open(test_img_file[i])
            img = img.resize((img_size[0], img_size[1]), Image.ANTIALIAS)
            pix_array = np.array(img)
            test_image.append(pix_array)
        test_image = np.array(test_image)
        self.test_image = test_image
        self.test_label = test_label
        self.transform = transform

    def __getitem__(self, index):
        img1, target1 = self.test_image[index], self.test_label[index]
        img1 = self.transform(img1)
        return img1, target1

    def __len__(self):
        return len(self.test_image)


def load_data(input_data_path):
    with open(input_data_path) as f:
        data_file_list = open(input_data_path, 'rt').read().splitlines()
        # Get full list of image and labels
        file_image = [s.split(' ')[0] for s in data_file_list]
        file_label = [int(s.split(' ')[1]) for s in data_file_list]

    return file_image, file_label
