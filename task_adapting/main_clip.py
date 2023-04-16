#!/usr/bin/env python3
from __future__ import print_function

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
import utils.logging as logging
from utils.functional import set_seed
from models import *
from adapter_clip import Adapter
from data_utils import loader as data_loader
from models import builder as model_builder
from launch import logging_train_setup



def load_dataset(args):
    """Load datasets for task adaption.
    """
    set_seed(args.seed)
    # load test
    trainset, class_names = data_loader.construct_train_loader(args, args.test_dataset)
    minis_test = [
        trainset, 
        data_loader.construct_val_loader(args, args.test_dataset)[0], 
        data_loader.construct_test_loader(args, args.test_dataset)[0]
    ]
    return minis_test, refine_classname(class_names)


def refine_classname(class_names):
    for i, class_name in enumerate(class_names):
        class_names[i] = class_name.lower().replace('_', ' ').replace('-', ' ')
    return class_names


def main():
    """Task Adaption on the downstream dataset.
    """
    assert args.pretrained_model.startswith("clip-")

    # load datasets for meta train or test
    minis_test, class_names = load_dataset(args)

    # load pretrained model
    model, cur_device = model_builder._construct_model(args)

    # initialize meta-learner
    metalearner = Adapter(args, model)
    metalearner.model.to(cur_device)

    # start task adapting
    if args.adapt_method == "prompt_wo_head":
        prompter_path = None if args.checkpoint_dir == "" else os.path.join(BASE_DIR, args.checkpoint_dir)
        accs = metalearner.our_method(minis_test, prompter_path, class_names)

    elif args.adapt_method == "prompt_w_head":
        raise NotImplementedError("The CLIP version of head-tuning version is not available here.")

    else:
        raise NotImplementedError





if __name__ == '__main__':
    args = Arguments(stage='task_adapting').parser().parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # setup training env including loggers
    logging_train_setup(args)
    logger = logging.get_logger("dam-vp")

    # basic configuration
    set_seed(args.seed)
    logger.info(f"Using random seed: {args.seed}")


    # main loop
    main()