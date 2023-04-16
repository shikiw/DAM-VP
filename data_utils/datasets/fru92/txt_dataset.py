#!/usr/bin/env python3

"""TXT dataset: support Fru92"""

# from macpath import split
import os
import sys
import torch
import torch.utils.data
import torchvision as tv
import numpy as np
from collections import Counter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, '../../../'))

from data_utils.transforms import get_transforms
from utils import logging
logger = logging.get_logger("dam-vp")



def read_txt(filename):
    """read txt files"""
    f = open(filename, encoding="utf-8")
    files, labels, classes = [], [], []
    for line in f:
        file, label = line.strip().split()
        class_name = file.split("/")[0]
        files.append(file)
        labels.append(label)
        classes.append(class_name)
    return files, labels, classes


class TXTDataset(torch.utils.data.Dataset):
    def __init__(self, args, split):
        assert split in {
            "train",
            "val",
            "test",
        }, "Split '{}' not supported for {} dataset".format(
            split, args.dataset)
        # logger.info("Constructing {} dataset {}...".format(
        #     args.dataset, split))

        self.args = args
        self._split = split
        self.data_dir = args.data_dir
        self._construct_imdb()
        self.transform = get_transforms(split, args.crop_size, args.pretrained_model)

    def get_anno(self):
        anno_path = os.path.join(self.data_dir, "fru92_lists/fru_{}.txt".format(self._split))
        assert os.path.exists(anno_path), "{} dir not found".format(anno_path)

        return read_txt(anno_path)

    def get_imagedir(self):
        raise NotImplementedError()

    def _construct_imdb(self):
        """Constructs the imdb."""

        img_dir = self.get_imagedir()
        assert os.path.exists(img_dir), "{} dir not found".format(img_dir)

        anno_file, anno_label, anno_classes = self.get_anno()
        # Map class ids to contiguous ids
        self._class_ids = sorted(list(set(anno_label)))
        self._class_id_cont_id = {v: i for i, v in enumerate(self._class_ids)}
        self.classes = sorted(list(set(anno_classes)))

        assert len(anno_file) == len(anno_label)
        # Construct the image db
        self._imdb = []
        for i in range(len(anno_file)):
            cont_id = self._class_id_cont_id[anno_label[i]]
            im_path = os.path.join(img_dir, anno_file[i])
            self._imdb.append({"im_path": im_path, "class": cont_id})

        logger.info("Nums of images: {}".format(len(self._imdb)))
        logger.info("Nums of classes: {}".format(len(self._class_ids)))

    def get_info(self):
        num_imgs = len(self._imdb)
        return num_imgs, self.get_class_num()

    def get_class_num(self):
        return self.args.num_classes
        # return len(self._class_ids)

    def get_class_weights(self, weight_type):
        """get a list of class weight, return a list float"""

        if "train" not in self._split:
            raise ValueError(
                "only getting training class distribution, " + \
                "got split {} instead".format(self._split)
            )

        cls_num = self.get_class_num()
        if weight_type == "none":
            return [1.0] * cls_num

        id2counts = Counter(self.get_anno()[1])
        assert len(id2counts) == cls_num
        num_per_cls = np.array([id2counts[i] for i in self._class_ids])

        if weight_type == 'inv':
            mu = -1.0
        elif weight_type == 'inv_sqrt':
            mu = -0.5
        weight_list = num_per_cls ** mu
        weight_list = np.divide(
            weight_list, np.linalg.norm(weight_list, 1)) * cls_num
        return weight_list.tolist()

    def __getitem__(self, index):
        # Load the image
        im = tv.datasets.folder.default_loader(self._imdb[index]["im_path"])
        label = self._imdb[index]["class"]
        im = self.transform(im)
        # if self._split == "train":
        #     index = index
        # else:
        #     index = f"{self._split}{index}"
        sample = {
            "image": im,
            "label": label,
            # "id": index
        }
        return sample

    def __len__(self):
        return len(self._imdb)


class Fru92Dataset(TXTDataset):
    """Fru92 dataset."""

    def __init__(self, args, split):
        super(Fru92Dataset, self).__init__(args, split)

    def get_imagedir(self):
        return os.path.join(self.data_dir, "fru92_images/")




if __name__ == '__main__':
    import argparse
    from torch.utils.data import DataLoader

    parser = argparse.ArgumentParser(description='Meta Training for Visual Prompts')
    parser.add_argument('--dataset', type=str, default="fru92")
    parser.add_argument('--data_dir', type=str, default="/data-x/g12/huangqidong/vegfru-dataset")
    parser.add_argument('--pretrained_model', type=str, default="vit-b-22k")
    parser.add_argument('--crop_size', type=int, default=224)
    parser.add_argument('--num_classes', type=int, default=92)
    args = parser.parse_args()

    dataset_train = Fru92Dataset(args, "train")
    dataset_val = Fru92Dataset(args, "val")
    dataset_test = Fru92Dataset(args, "test")
    logger.info(dataset_train.classes[0])
    logger.info("Nums of classes: {}".format(len(dataset_train.classes)))
    logger.info("Sample nums: [train]-{}, [val]-{}, [test]-{}".format(len(dataset_train), len(dataset_val), len(dataset_test)))

    for sample in DataLoader(dataset_train, batch_size=32):
        logger.info(sample["image"].shape)
        logger.info(sample["label"].shape)
        break



