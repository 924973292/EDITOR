from __future__ import division, print_function, absolute_import
import glob
import warnings
import os.path as osp
from .bases import BaseImageDataset


class RegDB(BaseImageDataset):

    def __init__(self, root='', verbose=True, **kwargs):
        super(RegDB, self).__init__()
        self.dataset_dir = osp.abspath(osp.expanduser(root))

        # allow alternative directory structure
        self.data_dir = self.dataset_dir
        data_dir = osp.join(self.data_dir)
        if osp.isdir(data_dir):
            self.data_dir = data_dir
        else:
            warnings.warn(
                'The current data structure is deprecated.'
            )

        self.train_dir = osp.join(self.data_dir, 'train')
        self.query_dir = osp.join(self.data_dir, 'test')
        self.gallery_dir = osp.join(self.data_dir, 'test')

        self._check_before_run()

        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)
        if verbose:
            print("=> RegDB loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(
            self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(
            self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(
            self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, relabel=False):
        img_paths_RGB = glob.glob(osp.join(dir_path, 'RGB', '*.bmp'))
        pid_container = set()
        for img_path_RGB in img_paths_RGB:
            bmp_name = img_path_RGB.split('/')[-1].split('.')[0]
            pid = int(bmp_name.split('_')[-1])
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        data = []
        for img_path_RGB in img_paths_RGB:
            img = []
            bmp_name = img_path_RGB.split('/')[-1]
            bmp_name = bmp_name.replace('v', 't')
            img_path_NI = osp.join(dir_path, 'NI', bmp_name)
            img.append(img_path_RGB)
            img.append(img_path_NI)
            bmp_name = bmp_name.split('.')[0]
            pid = int(bmp_name.split('_')[-1])
            camid = 1
            trackid = -1
            if relabel:
                pid = pid2label[pid]
            data.append((img, pid, camid, trackid))
            # print("11111")
        return data
