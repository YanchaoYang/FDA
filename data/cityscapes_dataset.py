import torch
import numpy as np
import os
import os.path as osp
from PIL import Image
from torch.utils import data

class cityscapesDataSet(data.Dataset):
    def __init__(self, root, list_path, crop_size=(11, 11), mean=(128, 128, 128), max_iters=None, set='val'):
        self.root = root
        self.list_path = list_path
        self.crop_size = crop_size
        self.mean = mean
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters==None:
            self.img_ids = self.img_ids * int( np.ceil(float(max_iters)/len(self.img_ids)) )
        self.files = []
        self.set = set

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, index):
        name = self.img_ids[index]
        image = Image.open( osp.join(self.root, "leftImg8bit/%s/%s" % (self.set, name)) ).convert('RGB')
        # resize
        image = image.resize( self.crop_size, Image.BICUBIC )
        image = np.asarray(image, np.float32)

        size = image.shape
        image = image[:, :, ::-1]  # change to BGR
        image -= self.mean
        image = image.transpose((2, 0, 1))

        return image.copy(), np.array(size), name

