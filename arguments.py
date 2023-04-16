# -*- coding: utf-8 -*-

import argparse
from sqlite3 import NotSupportedError


MODEL_LIST = [
    "resnet50-1k", 
    "vit-b-1k", 
    "vit-b-22k", 
    "swin-b-1k", 
    "swin-b-22k", 
    "moco-v3-b-1k",
    "clip-resnet50", 
    "clip-vit-b"
]

META_TRAIN_DATASETS = [
    "sun397", 
    "stl10", 
    "fru92", 
    "veg200", 
    "oxford-iiit-pets", 
    "eurosat"
]

TASK_ADAPT_DATASETS = [
    # original setting
    "cifar10", 
    "cifar100", 
    "cub200",  
    "nabirds", 
    "oxford-flowers", 
    "stanford-dogs", 
    "stanford-cars", 
    "fgvc-aircraft", 
    "food101", 
    "dtd", 
    "svhn", 
    "gtsrb",
    # vtab benchmark
    "vtab-caltech101",
    "vtab-cifar(num_classes=100)",
    "vtab-dtd",
    "vtab-oxford_flowers102",
    "vtab-oxford_iiit_pet",
    "vtab-patch_camelyon",
    "vtab-sun397",
    "vtab-svhn",
    "vtab-resisc45",
    "vtab-eurosat",
    "vtab-dmlab",
    "vtab-kitti(task=\"closest_vehicle_distance\")",
    "vtab-smallnorb(predicted_attribute=\"label_azimuth\")",
    "vtab-smallnorb(predicted_attribute=\"label_elevation\")",
    "vtab-dsprites(predicted_attribute=\"label_x_position\",num_classes=16)",
    "vtab-dsprites(predicted_attribute=\"label_orientation\",num_classes=16)",
    "vtab-clevr(task=\"closest_object_distance\")",
    "vtab-clevr(task=\"count_all\")",
    "vtab-diabetic_retinopathy(config=\"btgraham-300\")"
]

DATASET_DIVERSITIES = {
    "cifar10": 70.2, 
    "cifar100": 70.9, 
    "cub200": 76,  
    "nabirds": 74.8, 
    "oxford-flowers": 72.7, 
    "stanford-dogs": 73.4, 
    "stanford-cars": 70.5, 
    "fgvc-aircraft": 65.9, 
    "food101": 72.7, 
    "dtd": 78.7, 
    "svhn": 61.8, 
    "gtsrb": 67.5,
    "sun397": 76.9, 
    "stl10": 74.1, 
    "fru92": 74.1, 
    "veg200": 71.5, 
    "oxford-iiit-pets": 72.4, 
    "eurosat": 64.6,
    # vtab benchmark
    "vtab-caltech101": None,
    "vtab-cifar(num_classes=100)": None,
    "vtab-dtd": None,
    "vtab-oxford_flowers102": None,
    "vtab-oxford_iiit_pet": None,
    "vtab-patch_camelyon": None,
    "vtab-sun397": None,
    "vtab-svhn": None,
    "vtab-resisc45": None,
    "vtab-eurosat": None,
    "vtab-dmlab": None,
    "vtab-kitti(task=\"closest_vehicle_distance\")": None,
    "vtab-smallnorb(predicted_attribute=\"label_azimuth\")": None,
    "vtab-smallnorb(predicted_attribute=\"label_elevation\")": None,
    "vtab-dsprites(predicted_attribute=\"label_x_position\",num_classes=16)": None,
    "vtab-dsprites(predicted_attribute=\"label_orientation\",num_classes=16)": None,
    "vtab-clevr(task=\"closest_object_distance\")": None,
    "vtab-clevr(task=\"count_all\")": None,
    "vtab-diabetic_retinopathy(config=\"btgraham-300\")": None
}


class Arguments:
    def __init__(self, stage='task_adapting'):
        self._parser = argparse.ArgumentParser(description='Diversity-Aware Meta Visual Prompting.')

        self.add_common_args()
        if stage == 'meta_training':
            self.add_meta_train_args()
        elif stage == "task_adapting":
            self.add_task_adapt_args()
        else:
            raise NotSupportedError

    def add_common_args(self):
        ### log related
        self._parser.add_argument('--output_dir', type=str, default='')

        ### data related
        self._parser.add_argument('--batch_size', type=int, default=128, help='Batch size in training')
        self._parser.add_argument('--base_dir', type=str, default='/data-x/g12/huangqidong/')
        self._parser.add_argument('--dataset_perc', default=1.0, type=float, help='Dataset percentage for usage [default: 1.0].')
        self._parser.add_argument('--crop_size', default=224, type=int, help='Input size of images [default: 224].')
        self._parser.add_argument('--diversities', type=dict, default=DATASET_DIVERSITIES, help='Diversity values of datasets.')

        ### prompt related
        self._parser.add_argument('--pretrained_model', type=str, default='vit-b-1k', choices=MODEL_LIST)
        self._parser.add_argument('--prompt_method', type=str, default='padding', choices=['padding', 'fixed_patch', 'random_patch'])
        self._parser.add_argument('--prompt_size', type=int, default=30, help='Padding size for visual prompts.')
        self._parser.add_argument('--wo_da', action='store_true', default=False, help='Without diversity-aware strategy [default: False].')

        ### model related
        self._parser.add_argument('--drop', type=float, default=0.0, metavar='PCT', help='Dropout rate (default: 0.)')
        self._parser.add_argument('--attn_drop_rate', type=float, default=0.0, metavar='PCT', help='Attention dropout rate (default: 0.)')
        self._parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT', help='Drop path rate (default: 0.1)')

        ### others
        self._parser.add_argument('--seed', type=int, default=2023, metavar='S', help='Random seed (default: 2023)')
        self._parser.add_argument('--gpu_ids', type=int, default=0, help='Ids of GPUs to use.')
        self._parser.add_argument('--num_gpus', type=int, default=1, help='Num of GPUs to use.')
        self._parser.add_argument('--num_workers', type=int, default=4, help='Worker nums of data loading.')
        self._parser.add_argument('--pin_memory', action='store_true', help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
        self._parser.set_defaults(pin_memory=True)
        self._parser.add_argument('--distributed', action='store_true', default=False, help='Whether to use the distributed mode [default: False].')


    def add_meta_train_args(self):
        ### data related
        self._parser.add_argument('--meta_datasets', type=list, default=META_TRAIN_DATASETS, help='The datasets selected for meta training')
        self._parser.add_argument('--test_dataset', type=str, default='oxford-flowers', choices=TASK_ADAPT_DATASETS, help='The dataset selected for evaluation')
        self._parser.add_argument('--adapt_method', type=str, default='prompt_wo_head', choices=['prompt_wo_head', 'prompt_w_head'])

        ### meta training related
        self._parser.add_argument('--epochs', type=int, default=200, help='Nums of training epochs.')
        self._parser.add_argument('--optimizer', type=str, default='Adam', help='Optimizer for training [default: Adam]')
        self._parser.add_argument('--num_tasks', type=int, default=6, help='Meta batch size, namely task num')
        self._parser.add_argument('--meta_lr', type=float, default=1.0, help='Meta-level outer learning rate')
        self._parser.add_argument('--update_lr', type=float, default=1.0, help='Task-level inner update learning rate')
        self._parser.add_argument('--weight_decay', type=float, default=1e-4, help='Task-level inner update weight decay rate')
        self._parser.add_argument('--update_step', type=int, default=8, help='Task-level inner update steps')
        self._parser.add_argument('--update_step_test', type=int, default=20, help='Update steps for finetunning')
        self._parser.add_argument('--meta_optim_choose', type=str, default="reptile", choices=['reptile'], help='Choice for using which meta learning method')
        self._parser.add_argument('--meta_step_size', type=float, default=1.0, help='Task-level outer update step size')


    def add_task_adapt_args(self):
        ### data related
        self._parser.add_argument('--test_dataset', type=str, default='oxford-flowers', choices=TASK_ADAPT_DATASETS, help='The dataset selected for evaluation')
        self._parser.add_argument('--adapt_method', type=str, default='prompt_wo_head', choices=['prompt_wo_head', 'prompt_w_head'])
        self._parser.add_argument('--checkpoint_dir', type=str, default='')

        ### tuning related
        self._parser.add_argument('--epochs', type=int, default=50, help='Nums of training epochs.')
        self._parser.add_argument('--optimizer', type=str, default='Adam', help='Optimizer for training [default: Adam]')
        self._parser.add_argument('--lr', type=float, default=1e+4, help='Task adapting learning rate')
        self._parser.add_argument('--weight_decay', type=float, default=0, help='Task adapting weight decay rate')
        self._parser.add_argument('--eval_only', action='store_true', default=False, help='Evaluate only [default: False].')


    def parser(self):
        return self._parser



    
   


