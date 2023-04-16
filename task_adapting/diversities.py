#!/usr/bin/env python3
from __future__ import print_function
from dis import dis

import os
from subprocess import check_output
import sys
import numpy as np
from tqdm import tqdm
from copy import deepcopy

import torch


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, '../'))

from arguments import Arguments
from utils.functional import set_seed
from models import *
from data_utils import loader as data_loader
import lpips


class util_of_lpips():
    def __init__(self, net, device):
        """Learned Perceptual Image Patch Similarity, LPIPS.
        https://github.com/richzhang/PerceptualSimilarity/blob/master/lpips_2imgs.py
        
        args:
            net: str, ['alex', 'vgg']
            use_gpu: bool
        """
        ## Initializing the model
        self.loss_fn = lpips.LPIPS(net=net)
        self.device = device
        self.loss_fn.to(device)


    def calc_lpips(self, img_batch1, img_batch2):
        """LPIPS distance calculator. 

        args:
            img_batch1 : tensor
            img_batch2 : tensor
        """
        img_batch1 = img_batch1.to(self.device)
        img_batch2 = img_batch2.to(self.device)
        dist = self.loss_fn.forward(img_batch1, img_batch2)
        return dist


def load_dataset(args):
    """Load datasets for task adaption.
    """
    # load test
    minis_test = [
        data_loader.construct_train_loader(args, args.test_dataset), 
        data_loader.construct_val_loader(args, args.test_dataset), 
        data_loader.construct_test_loader(args, args.test_dataset)
    ]
    return minis_test

@torch.no_grad()
def main():
    """Task Adaption on the downstream dataset.
    """

    # load datasets for diversity calculation
    minis_test = data_loader.construct_train_loader(args, args.test_dataset)

    # introduce LPIPS
    lpips_func = util_of_lpips(net="alex", device=args.device)

    # randomly select and obtain the diversity using average lpips
    dist_total = 0
    num_total = 0
    for i, sample in enumerate(minis_test):
        image = sample["image"].to(args.device)
        order = torch.randperm(image.size(0))
        image_shuffled = image[order]

        dist = lpips_func.calc_lpips(image.detach(), image_shuffled.detach())
        dist = dist[dist != 0]
        dist_total += dist.sum()
        num_total += dist.size(0)
        print(dist.sum().item()/dist.size(0))
        if i == 1000 // args.batch_size - 1:
            break

    dist_total /= num_total
    print("Data diversity score of {}: {}".format(args.test_dataset, dist_total))



if __name__ == '__main__':
    args = Arguments(stage='task_adapting').parser().parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # basic configuration
    set_seed(args.seed)

    # main loop
    main()