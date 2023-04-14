import os
import sys
import json
import random
from pathlib import Path
from typing import Any, Tuple, Callable, Optional

import PIL.Image

from torchvision.datasets.utils import verify_str_arg, download_and_extract_archive
from torchvision.datasets.vision import VisionDataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, '../../../'))

from data_utils.transforms import get_transforms
from utils import logging
logger = logging.get_logger("dam-vp")


class Food101(VisionDataset):
    """`The Food-101 Data Set <https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/>`_.

    The Food-101 is a challenging data set of 101 food categories, with 101'000 images.
    For each class, 250 manually reviewed test images are provided as well as 750 training images.
    On purpose, the training images were not cleaned, and thus still contain some amount of noise.
    This comes mostly in the form of intense colors and sometimes wrong labels. All images were
    rescaled to have a maximum side length of 512 pixels.


    Args:
        root (string): Root directory of the dataset.
        split (string, optional): The dataset split, supports ``"train"`` (default) and ``"test"``.
        transform (callable, optional): A function/transform that  takes in an PIL image and returns a transformed
            version. E.g, ``transforms.RandomCrop``.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again. Default is False.
    """

    _URL = "http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz"
    _MD5 = "85eeb15f3717b99a5da872d97d918f87"

    def __init__(
        self,
        # root: str,
        args,
        split: str = "train",
        percentage: float = 0.8,
        # transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = True,
    ) -> None:
        super().__init__(
            args.data_dir, 
            transform=get_transforms(split, args.crop_size, args.pretrained_model), 
            target_transform=target_transform
        )
        self._split = verify_str_arg(split, "split", ("train", "val", "test"))
        self._base_folder = Path(self.root) / "food-101"
        self._meta_folder = self._base_folder / "meta"
        self._images_folder = self._base_folder / "images"

        if download:
            self._download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        self._labels = []
        self._image_files = []
        split_name = "test" if split == "test" else "train"
        with open(self._meta_folder / f"{split_name}.json") as f:
            metadata = json.loads(f.read())

        self.classes = sorted(metadata.keys())
        self.class_to_idx = dict(zip(self.classes, range(len(self.classes))))

        for class_label, im_rel_paths in metadata.items():
            self._labels += [self.class_to_idx[class_label]] * len(im_rel_paths)
            self._image_files += [
                self._images_folder.joinpath(*f"{im_rel_path}.jpg".split("/")) for im_rel_path in im_rel_paths
            ]

        if split in ["train", "val"]:
            data_zip = list(zip(self._labels, self._image_files))
            random.shuffle(data_zip)
            self._labels[:], self._image_files[:] = zip(*data_zip)
            del data_zip

        if split == "train":
            self._labels = self._labels[:int(percentage*len(self._labels))]
            self._image_files = self._image_files[:int(percentage*len(self._image_files))]
        if split == "val":
            self._labels = self._labels[int(percentage*len(self._labels)):]
            self._image_files = self._image_files[int(percentage*len(self._image_files)):]

    def __len__(self) -> int:
        return len(self._image_files)

    def __getitem__(self, idx) -> Tuple[Any, Any]:
        image_file, label = self._image_files[idx], self._labels[idx]
        image = PIL.Image.open(image_file).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        sample = {
            "image": image,
            "label": label
        }
        return sample

    def extra_repr(self) -> str:
        return f"split={self._split}"

    def _check_exists(self) -> bool:
        return all(folder.exists() and folder.is_dir() for folder in (self._meta_folder, self._images_folder))

    def _download(self) -> None:
        if self._check_exists():
            return
        download_and_extract_archive(self._URL, download_root=self.root, md5=self._MD5)


if __name__ == '__main__':
    import argparse
    from torch.utils.data import DataLoader

    parser = argparse.ArgumentParser(description='Meta Training for Visual Prompts')
    parser.add_argument('--dataset', type=str, default="food101")
    parser.add_argument('--data_dir', type=str, default="/data-x/g12/huangqidong/")
    parser.add_argument('--pretrained_model', type=str, default="vit-b-22k")
    parser.add_argument('--crop_size', type=int, default=224)
    parser.add_argument('--num_classes', type=int, default=101)
    args = parser.parse_args()

    

    dataset_train = Food101(args, "train")
    dataset_val = Food101(args, "val")
    dataset_test = Food101(args, "test")
    logger.info(dataset_train.classes[0])
    logger.info("Number of classes: {}".format(len(dataset_train.classes)))
    logger.info("Sample nums: [train]-{}, [val]-{}, [test]-{}".format(len(dataset_train), len(dataset_val), len(dataset_test)))

    for sample in DataLoader(dataset_train, batch_size=32):
        logger.info(sample["image"].shape)
        logger.info(sample["label"].shape)
        break