import cv2
import numpy as np
import skimage
import os
def resize(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cur_ratio = img.shape[1] / float(img.shape[0])
    target_ratio = 128 / float(32)
    mask_h = 32
    mask_w = 128
    img = np.array(img)
    if cur_ratio > target_ratio:
        cur_h = 32
        cur_w = 128
    else:
        cur_h = 32
        cur_w = int(32 * cur_ratio)
    img = cv2.resize(img, (cur_w, cur_h))
    start_y = (mask_h - img.shape[0]) // 2
    start_x = (mask_w - img.shape[1]) // 2
    mask = np.zeros([mask_h, mask_w, 3]).astype(np.uint8)
    mask[start_y: start_y + img.shape[0], start_x: start_x + img.shape[1], :] = img
    return mask

img = skimage.io.imread(os.path.join(img_path, img_name))
img_new = resize(img)
amap = cv2.cvtColor(skimage.io.imread(os.path.join(path, att_map)), cv2.COLOR_RGB2BGR)
new_map = cv2.resize(amap, (img_new.shape[1], img_new.shape[0]))
normed_mask = new_map / np.max(new_map)
normed_mask = np.uint8(255 * normed_mask)
normed_mask = cv2.applyColorMap(normed_mask, cv2.COLORMAP_JET)
normed_mask = cv2.addWeighted(img_new, 0.6, normed_mask, 1.0, 0)
skimage.io.imsave(os.path.join(res_dir, m), cv2.cvtColor(normed_mask, cv2.COLOR_BGR2RGB))
