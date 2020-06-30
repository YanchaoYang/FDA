import os
import os.path as osp
from PIL import Image
import numpy as np
import torch
from torch.utils import data

class GTA5DataSet(data.Dataset):

    def __init__(self, root, list_path, crop_size=(11, 11), resize=(11, 11), ignore_label=255, mean=(128, 128, 128), max_iters=None):
        self.root = root  # folder for GTA5 which contains subfolder images, labels
        self.list_path = list_path   # list of image names
        self.crop_size = crop_size   # dst size for resize
        self.resize = resize
        self.ignore_label = ignore_label
        self.mean = mean
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters==None:
            self.img_ids = self.img_ids * int(  np.ceil(float(max_iters)/len(self.img_ids))  )

        self.files = []

        self.id_to_trainid = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
                              19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                              26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}
    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, index):
        name = self.img_ids[index]
        image = Image.open(osp.join(self.root, "images/%s" % name)).convert('RGB')
        label = Image.open(osp.join(self.root, "labels/%s" % name))
        # resize
        image = image.resize(self.resize, Image.BICUBIC)
        label = label.resize(self.resize, Image.NEAREST)

        # (left, upper, right, lower)
        left = self.resize[0]-self.crop_size[0]
        upper= self.resize[1]-self.crop_size[1]

        left = np.random.randint(0, high=left)
        upper= np.random.randint(0, high=upper)
        right= left + self.crop_size[0]
        lower= upper+ self.crop_size[1]

        image = image.crop((left, upper, right, lower))
        label = label.crop((left, upper, right, lower))

        image = np.asarray(image, np.float32)
        label = np.asarray(label, np.float32)

        label_copy = self.ignore_label * np.ones(label.shape, dtype=np.float32)
        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v

        size = image.shape
        image = image[:, :, ::-1]  # change to BGR
        image -= self.mean
        image = image.transpose((2, 0, 1))

        return image.copy(), label_copy.copy(), np.array(size), name

