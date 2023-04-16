import os
from statistics import mode
import sys
from tkinter.messagebox import NO
import numpy as np
import pandas as pd 
import os.path as osp
from copy import deepcopy
import clip

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import sklearn.cluster as cluster

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, '../'))

from models import prompters
from data_utils import loader as data_loader
from data_utils.datasets import *
from utils.functional import set_seed
from utils.train_utils import cosine_lr
import utils.logging as logging


logger = logging.get_logger("dam-vp")
class Adapter(object):
    """A Gather of Our Task Adapting Methods.
    """

    def __init__(self, args, model):
        super(Adapter, self).__init__()
        self.args = args
        self.model = model.eval()

        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.num_classes = data_loader._dataset_class_num(self.args.test_dataset)
        self.criterion = torch.nn.CrossEntropyLoss().to(args.device)


    def nums_of_learnable_params(self, model):
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info('number of params (M): %.2f' % (n_parameters / 1.e6))


    def loss_function(self, logits, target):
        """Loss function to predict GT target.
        """
        loss = self.criterion(logits, target)
        return loss


    def zero_shot_transfer(self, test_data, class_names):
        """Zero-Shot Transfer based on CLIP.
        """
        _, _, test_loader = test_data
        num_classes = data_loader._dataset_class_num(self.args.test_dataset)
        text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") \
            for c in class_names]).to(self.args.device)
        logger.info(class_names)
        assert text_inputs.size(0) == num_classes

        # test
        with torch.no_grad():
            num_total, correct = 0, 0
            for i, sample in enumerate(test_loader):
                image = sample["image"].to(self.args.device)
                label = sample["label"].to(self.args.device)
                logits = self.model(image, text_inputs)[0]
                pred = torch.argmax(logits, dim=-1)
                correct += (pred == label).sum().item()
                num_total += image.size(0)
            acc_test = float(correct / num_total)
            logger.info("[CLIP Zero-Shot Testing] Test acc: {}".format(acc_test))
        return acc_test


    def load_prompter(self, prompter_path=None):
        """Load the trained visual prompter.
        """
        prompter = prompters.__dict__[self.args.prompt_method](self.args).to(self.args.device)
        # for k, v in prompter.named_parameters():
        #     v.data.zero_()
        if prompter_path is not None:
            checkpoint = torch.load(prompter_path)
            prompter.load_state_dict(checkpoint['state_dict'])
        return prompter


    def get_prompted_image(self, image, text_inputs=None, prototype_gather=None, prompter=None, prompter_gather=None):
        """Obtain the prompted batch images.
        """
        if self.args.wo_da:
            assert prompter is not None
            prompted_image = prompter(image)
        else:
            assert prototype_gather is not None
            assert prompter_gather is not None
            prompted_image = []
            with torch.no_grad():
                # rep_batch = self.model.encode_image(image) # [N, emd_dim]
                rep_batch = self.model(image, text_inputs)[0] # [N, emd_dim]
                rep_batch_sum = (rep_batch**2).sum(dim=-1, keepdims=True) # [N, 1]
                prototype_gather_sum = (prototype_gather**2).sum(dim=-1, keepdims=True).T # [1, M]
                distance_matrix = torch.sqrt(rep_batch_sum + prototype_gather_sum - 2 * torch.mm(rep_batch, prototype_gather.T)) # [N, M]
                indices = torch.argmin(distance_matrix, dim=-1) # [B]

            for idx in range(rep_batch.size(0)):
                prompted_image.append(
                    prompter_gather[indices[idx]](image[idx].unsqueeze(0))
                )
            prompted_image = torch.cat(prompted_image, dim=0)
        return prompted_image


    def coarse_clustering(self, data_loader, text_inputs):
        """Diversity-Aware Adaption on downstream data.
        We roughly divide the downstream task data into several partitions, each 
        partition presents a coarsely divided cluster. Different clusters correspond 
        to different prompters. 
        """
        TASK_ADAPT_DATASETS = {
            "cifar10": 6, 
            "cifar100": 16, 
            "cub200": 40,  
            "nabirds": 75, 
            "oxford-flowers": 25, 
            "stanford-dogs": 25, 
            "stanford-cars": 40, 
            "fgvc-aircraft": 25, 
            "food101": 40, 
            "dtd": 16, 
            "svhn": 16, 
            "gtsrb": 16
        }
        train_loader, _, _ = data_loader
        hc = cluster.AgglomerativeClustering(
            n_clusters=None, 
            linkage='average', 
            distance_threshold=TASK_ADAPT_DATASETS[self.args.test_dataset]
        )
        with torch.no_grad():
            for i, sample in enumerate(train_loader):
                image = sample["image"].to(self.args.device)
                # rep = self.model.encode_image(image)
                rep = self.model(image, text_inputs)[0]
                if i < 1:
                    rep_gather = rep
                else:
                    rep_gather = torch.cat([rep_gather, rep], dim=0)

                if rep_gather.size(0) > 1000:
                    rep_gather = rep_gather[:1000]
                    break

        y_pred = hc.fit(rep_gather.detach().cpu().numpy()).labels_
        y_pred = torch.from_numpy(y_pred).to(self.args.device)
        coarse_class_idx = torch.unique(y_pred)
        self.num_coarse_classes = len(coarse_class_idx)
        logger.info("Nums of coarsely divided categories for test dataset {}: {}".format(
            self.args.test_dataset, len(coarse_class_idx)))

        prototype_gather = []
        for i in range(len(coarse_class_idx)):
            pos = torch.where(y_pred == i)[0]
            prototype = rep_gather[pos].mean(0).unsqueeze(0)
            prototype_gather.append(prototype)
        self.prototype_gather = torch.cat(prototype_gather)
        logger.info("Nums of prototypes of coarse clusters for test dataset {}: {}".format(
            self.args.test_dataset, self.prototype_gather.size(0)))


    def our_method(self, test_data, prompter_path, class_names):
        """Diversity-Aware Meta Visual Prompting.
        """
        train_loader, val_loader, test_loader = test_data
        num_classes = data_loader._dataset_class_num(self.args.test_dataset)
        logger.info(class_names[0])
        text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") \
            for c in class_names]).to(self.args.device)
        assert text_inputs.size(0) == num_classes

        prompter = self.load_prompter(prompter_path)
        if not self.args.wo_da:
            self.coarse_clustering(test_data, text_inputs)

        if self.args.wo_da:
            # prompter = deepcopy(prompter)
            optimizer = torch.optim.SGD(
                prompter.parameters(),
                lr=self.lr,
                momentum=0.9,
                weight_decay=self.weight_decay
            )
        else:
            prompter_gather, prompter_params_gather = [], []
            for i in range(self.num_coarse_classes):
                prompter_gather.append(
                    deepcopy(prompter)
                )
                prompter_params_gather.append(
                    {'params':prompter_gather[i].parameters()}
                )
            optimizer = torch.optim.SGD(
                prompter_params_gather,
                lr=self.lr,
                momentum=0.9,
                weight_decay=self.weight_decay
            )
        scheduler = cosine_lr(
            optimizer, 
            self.lr, 
            len(train_loader)*self.args.epochs//5, 
            len(train_loader)*self.args.epochs
        )

        BEST_ACC_VAL = -np.inf
        if self.args.wo_da:
            best_prompter = deepcopy(prompter)
        else:
            best_prompter_gather = deepcopy(prompter_gather)

        for epoch in range(self.args.epochs):
            # train
            for i, sample in enumerate(train_loader):
                # adjust learning rate
                global_step = len(train_loader) * epoch + i
                scheduler(global_step)
                image = sample["image"].to(self.args.device)
                label = sample["label"].to(self.args.device)
                prompted_image = self.get_prompted_image(image, prompter=prompter) \
                    if self.args.wo_da else self.get_prompted_image(image, text_inputs, self.prototype_gather, prompter_gather=prompter_gather)
                logits = self.model(prompted_image, text_inputs)[0]
                # logger.info(logits.max())
                loss = self.loss_function(logits, label)
                optimizer.zero_grad()
                loss.backward()
                # logger.info(prompter_gather[0].pad_up.grad)
                optimizer.step()

                if (i + 1) % 1 == 0:
                    logger.info("[Prompt Finetuning] Epoch: [{}/{}], Step: [{}/{}], Training loss: {}".format(
                        epoch, self.args.epochs, i, len(train_loader), loss.item()))
            # validate
            with torch.no_grad():
                num_total, correct = 0, 0
                for i, sample in enumerate(val_loader):
                    image = sample["image"].to(self.args.device)
                    label = sample["label"].to(self.args.device)
                    prompted_image = self.get_prompted_image(image, prompter=prompter) \
                        if self.args.wo_da else self.get_prompted_image(image, text_inputs, self.prototype_gather, prompter_gather=prompter_gather)
                    logits = self.model(prompted_image, text_inputs)[0]
                    pred = torch.argmax(logits, dim=-1)
                    correct += (pred == label).sum().item()
                    num_total += image.size(0)
                acc_val = float(correct / num_total)
                logger.info("[Prompt Validating] Epoch: {}, Val acc: {}".format(epoch, acc_val))
                if acc_val > BEST_ACC_VAL:
                    BEST_ACC_VAL = acc_val
                    if self.args.wo_da:
                        best_prompter = deepcopy(prompter)
                    else:
                        best_prompter_gather = deepcopy(prompter_gather)
        # test
        with torch.no_grad():
            num_total, correct = 0, 0
            for i, sample in enumerate(test_loader):
                image = sample["image"].to(self.args.device)
                label = sample["label"].to(self.args.device)
                prompted_image = self.get_prompted_image(image, prompter=best_prompter) \
                    if self.args.wo_da else self.get_prompted_image(image, text_inputs, self.prototype_gather, prompter_gather=best_prompter_gather)
                logits = self.model(prompted_image, text_inputs)[0]
                pred = torch.argmax(logits, dim=-1)
                correct += (pred == label).sum().item()
                num_total += image.size(0)
            acc_test = float(correct / num_total)
            logger.info("[Prompt Testing] Epoch: {}, Test acc: {}".format(epoch, acc_test))
        return acc_test

