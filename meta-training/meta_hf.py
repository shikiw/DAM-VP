import os
from statistics import mode
import sys
from tkinter.messagebox import NO
import numpy as np
import pandas as pd 
import os.path as osp
from copy import deepcopy

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
from utils.functional import set_seed
from utils.train_utils import cosine_lr, _warmup_lr
import utils.logging as logging


logger = logging.get_logger("dam-vp")
class Meta(nn.Module):
    """Diversity-Aware Prompt Initilization Based on Fast Meta Learning.
    """
    def __init__(self, args, model):
        super(Meta, self).__init__()
        self.args = args
        self.model = model.eval()
        self.model.discard_classifier()

        # self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.num_tasks = args.num_tasks
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test
        self.meta_optim_choose = args.meta_optim_choose

        self.indices = self.get_active_neuron_index()
        self.prompter = prompters.__dict__[args.prompt_method](args).to(args.device)
        # self.meta_optimizer = optim.Adam(self.prompter.parameters(), lr=self.meta_lr)
        # self.meta_scheduler = optim.lr_scheduler.StepLR(self.meta_optimizer, step_size=20, gamma=0.7)
        self.criterion = torch.nn.CrossEntropyLoss().to(args.device)


    def lr_scheduler(self, step, steps, warmup_length):
        """Warm-start cosine scheduler for meta training.
        """
        if step < warmup_length:
            lr = _warmup_lr(self.meta_lr, warmup_length, step)
        else:
            e = step - warmup_length
            es = steps - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * self.meta_lr
        self.update_lr = lr


    def get_active_neuron_index(self):
        """Simulate the most active neurons in the representaion.
        """
        set_seed(self.args.seed)
        with torch.no_grad():
            input = torch.randn(512, 3, self.args.crop_size, self.args.crop_size).to(self.args.device)
            output = self.model.forward_features(input) # [512, emd_dim]
            output = output.std(0, unbiased=False) # [emd_dim]
            indices = output.sort(0, descending=True)[1]
        return indices


    def rep2logit(self, output, num_classes):
        """Convert output representation to logits
        """
        # activity aware
        indices = self.indices[:num_classes]
        indices = torch.unique(indices)
        logits = output[:, indices]
        return logits


    def loss_function(self, logits, target):
        """Loss function to predict GT target.
        """
        loss = self.criterion(logits, target)
        return loss


    def net_to_weights(self, net):
        """Turn the network to weight list.
        """
        weights = []
        for each in net.parameters():
            pp = torch.autograd.Variable(each.clone(), requires_grad=True)
            weights.append(pp)
        return weights


    def weights_to_net(self, weights):
        """Turn the weight list to the net.
        """
        net = deepcopy(self.net)
        dic = net.state_dict()
        keys = list(dic.keys())
        for i, each in enumerate(weights):
            dic[keys[i]] = each
        net.load_state_dict(dic)
        return net


    def computer_meta_target(self, weights_gather):
        """Get Meta target for Reptile.
        Calculate the target for meta-learner to get its meta-gradients.
        """
        meta_target = deepcopy(self.net)
        dic = meta_target.state_dict()
        keys = list(dic.keys())
        for i, each in enumerate(weights_gather[0]):
            pp = torch.zeros_like(each)
            for j in range(len(weights_gather)):
                pp += weights_gather[j][i]
            pp /= len(weights_gather)
            dic[keys[i]] = pp
        meta_target.load_state_dict(dic)
        return meta_target


    def computer_diff(self, prompter=None, prompter_gather=None):
        """This method is to computer the diifference between received fast prompter
        and the self.prompter.parameters, it will return the difference.
        """
        # dis = []
        # for each, fast_each in zip(self.net.parameters(), fast_net.parameters()):
        #     dis.append((each-fast_each).clone().detach())
        # return dis
        if self.args.wo_da:
            assert prompter is not None
            dis = []
            dic = self.prompter.state_dict()
            fast_dic = prompter.state_dict()
            for key in list(dic.keys()):
                dis.append((dic[key]-fast_dic[key]).clone().detach())
        else:
            assert prompter_gather is not None
            dis = []
            dic = self.prompter.state_dict()
            fast_dic_list = [prompter_gather[i].state_dict() for i in range(len(prompter_gather))]
            for key in list(dic.keys()):
                for i in range(len(fast_dic_list)):
                    if i < 1:
                        fast_dic_key = fast_dic_list[i][key]
                    else:
                        fast_dic_key += fast_dic_list[i][key]
                dis.append((len(fast_dic_list)*dic[key]-fast_dic_key).clone().detach())
        return dis


    def computer_meta_weight(self, task_diff_weights):
        """Meta optim for reptile.
        This method will update the self.net.parameters according to the reveived 
        weight difference gather from all tasks, which is the updating directions.
        The update learning rate is self.update.lr.
        """
        dic = self.prompter.state_dict()
        keys = list(dic.keys())
        for i, each in enumerate(task_diff_weights[0]):
            diff = torch.zeros_like(each).to(each.device)
            for j in range(len(task_diff_weights)):
                diff += task_diff_weights[j][i]
            diff = torch.true_divide(diff, len(task_diff_weights)) \
                if self.args.wo_da else torch.true_divide(diff, sum(self.num_coarse_classes_meta))
            diff = torch.tensor(diff, dtype=dic[keys[i]].dtype)
            diff = diff.to(dic[keys[i]].device)
            dic[keys[i]] -= self.args.meta_step_size * diff
        self.prompter.load_state_dict(dic)


    def forward(self, batch_data):
        """Meta training procedure.
        """
        task_num = len(batch_data)
        _, C, H, W = batch_data[0]["image"].size()

        losses_q = [0 for _ in range(self.update_step + 1)] # losses_q[i] is the loss on step i
        corrects = [0 for _ in range(self.update_step + 2)] # corrects[i] is the loss sum on step i
        querysz = 1 

        task_diff_weights = [] # task diff weights is the list to store all weights diffs in all tasks
        for task_idx in range(task_num):
            sample = batch_data[task_idx]
            x_spt = sample["image"].reshape([-1,C,H,W]).to(self.args.device) # [B, C, H, W]
            y_spt = sample["label"].reshape([-1]).to(self.args.device) # [B, num_classes]
            num_classes = data_loader._dataset_class_num(self.args.meta_datasets[task_idx])

            # set fast net from copied weights from current meta-learner
            if self.args.wo_da:
                fast_prompter = deepcopy(self.prompter)

                # the optimizer for each sub-task
                if self.meta_optim_choose == "reptile":
                    cur_task_optim = optim.Adam(fast_prompter.parameters(), lr = self.update_lr)

                # if self.meta_optim_choose == "maml":
                #     cur_task_optim = optim.SGD(fast_prompter.parameters(), lr = self.update_lr)

            else:
                fast_prompter_gather, fast_prompter_params_gather = [], []
                for i in range(self.num_coarse_classes_meta[task_idx]):
                    fast_prompter_gather.append(deepcopy(self.prompter))
                    fast_prompter_params_gather.append({'params':fast_prompter_gather[i].parameters()})

                # the optimizer for each sub-task
                if self.meta_optim_choose == "reptile":
                    cur_task_optim = optim.Adam(fast_prompter_params_gather, lr = self.update_lr)

                # if self.meta_optim_choose == "maml":
                #     cur_task_optim = optim.SGD(fast_prompter_params_gather, lr = self.update_lr)

            # the first update
            prompted_x_spt = self.get_prompted_image(x_spt, prompter=fast_prompter) \
                if self.args.wo_da else self.get_prompted_image(x_spt, self.prototype_gather_meta[task_idx], \
                prompter_gather=fast_prompter_gather)
            output = self.model.forward_features(prompted_x_spt)
            logits = self.rep2logit(output, num_classes)
            loss = self.loss_function(logits, y_spt)
            cur_task_optim.zero_grad()
            loss.backward()
            cur_task_optim.step()

            # this is the loss and accuracy before first update
            with torch.no_grad():
                prompted_x_spt_q = self.get_prompted_image(x_spt, prompter=self.prompter) \
                    if self.args.wo_da else self.get_prompted_image(x_spt, self.prototype_gather_meta[task_idx], \
                    prompter_gather=[self.prompter for _ in range(self.num_coarse_classes_meta[task_idx])])
                output_q = self.model.forward_features(prompted_x_spt_q)
                logits_q = self.rep2logit(output_q, num_classes)
                loss_q = self.loss_function(logits_q, y_spt)
                losses_q[0] += loss_q
                corrects[0] += loss_q.sum()

            # this is the loss and accuracy after the first update
            with torch.no_grad():
                prompted_x_spt_q = self.get_prompted_image(x_spt, prompter=fast_prompter) \
                    if self.args.wo_da else self.get_prompted_image(x_spt, self.prototype_gather_meta[task_idx], \
                    prompter_gather=fast_prompter_gather)
                output_q = self.model.forward_features(prompted_x_spt_q)
                logits_q = self.rep2logit(output_q, num_classes)
                loss_q = self.loss_function(logits_q, y_spt)
                losses_q[1] += loss_q
                corrects[1] += loss_q.sum()

            # continue the update loop
            for k in range(1, self.update_step):
                prompted_x_spt = self.get_prompted_image(x_spt, prompter=fast_prompter) \
                    if self.args.wo_da else self.get_prompted_image(x_spt, self.prototype_gather_meta[task_idx], \
                    prompter_gather=fast_prompter_gather)
                output = self.model.forward_features(prompted_x_spt)
                logits = self.rep2logit(output, num_classes)
                loss = self.loss_function(logits, y_spt)
                cur_task_optim.zero_grad()
                loss.backward()
                cur_task_optim.step()

                with torch.no_grad():
                    prompted_x_spt_q = self.get_prompted_image(x_spt, prompter=fast_prompter) \
                        if self.args.wo_da else self.get_prompted_image(x_spt, self.prototype_gather_meta[task_idx], \
                        prompter_gather=fast_prompter_gather)
                    output_q = self.model.forward_features(prompted_x_spt_q)
                    logits_q = self.rep2logit(output_q, num_classes)
                    loss_q = self.loss_function(logits_q, y_spt)
                    correct = loss_q.sum()
                    corrects[k + 1] = corrects[k + 1] + correct

            # get weight diff list and put it into task diff gather.
            current_task_diff = self.computer_diff(prompter=fast_prompter) \
                if self.args.wo_da else self.computer_diff(prompter_gather=fast_prompter_gather)
            task_diff_weights.append(current_task_diff)

            # # for maml, the net update chain is neccessary.
            # if self.meta_optim_choose == "maml":
            #     fast_weights = list(map(lambda p: p[1] - p[0], zip(current_task_diff, self.net.parameters())))
            #     temp_net = self.weights_to_net(fast_weights)

            #     output_q = temp_net(x_qry)
            #     loss_q = self.loss_function(output_q, y_qry)
            #     losses_q[k + 1] += loss_q

        # end of all tasks
        # sum over all losses on query set across all tasks
        # if self.meta_optim_choose == "maml":
        #     loss_q = losses_q[-1] / task_num
        #     # optimize theta parameters
        #     self.meta_optim.zero_grad()
        #     loss_q.backward()
        #     self.meta_optim.step()

        if self.meta_optim_choose == "reptile":
            self.computer_meta_weight(task_diff_weights)

        # evaluate
        for task_idx in range(task_num):
            sample = batch_data[task_idx]
            x_spt = sample["image"].reshape([-1,C,H,W]).to(self.args.device) # [B, C, H, W]
            y_spt = sample["label"].reshape([-1]).to(self.args.device) # [B, num_classes]
            num_classes = data_loader._dataset_class_num(self.args.meta_datasets[task_idx])

            with torch.no_grad():
                prompted_x_spt_q = self.get_prompted_image(x_spt, prompter=self.prompter) \
                    if self.args.wo_da else self.get_prompted_image(x_spt, self.prototype_gather_meta[task_idx], \
                    prompter_gather=[self.prompter for _ in range(self.num_coarse_classes_meta[task_idx])])
                output_q = self.model.forward_features(prompted_x_spt_q)
                logits_q = self.rep2logit(output_q, num_classes)
                loss_q = self.loss_function(logits_q, y_spt)
                corrects[k + 2] += loss_q.sum()
            

        for i in range(len(corrects)):
            corrects[i] = corrects[i].item()

        accs = np.array(corrects) / (querysz * task_num)
        return accs


    def coarse_clustering(self, data_loader, mode="task_adapting"):
        """Diversity-Aware Adaption on downstream data.
        We roughly divide the downstream task data into several partitions, each 
        partition presents a coarsely divided cluster. Different clusters correspond 
        to different prompters. 
        """
        threshold_dict = {
            "resnet50-1k": 21, 
            "vit-b-1k": 31, 
            "vit-b-22k": 10, 
            "swin-b-22k": 20, 
            "moco-v3-b-1k":18
        }
        if mode == "meta_training":
            task_num = len(data_loader)
            num_coarse_classes_meta, prototype_gather_meta = [], []
            hc = cluster.AgglomerativeClustering(
                n_clusters=None, 
                linkage='average', 
                distance_threshold=threshold_dict[self.args.pretrained_model]
            )
            for task_idx in range(task_num):
                with torch.no_grad():
                    for i, sample in enumerate(data_loader[task_idx]):
                        image = sample["image"].to(self.args.device)
                        rep = self.model.forward_features(image)
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
                num_coarse_classes_meta.append(len(coarse_class_idx))
                logger.info("Nums of coarsely divided categories for meta train dataset {}: {}".format(
                    self.args.meta_datasets[task_idx], len(coarse_class_idx)))

                prototype_gather = []
                for i in range(len(coarse_class_idx)):
                    pos = torch.where(y_pred == i)[0]
                    prototype = rep_gather[pos].mean(0).unsqueeze(0)
                    prototype_gather.append(prototype)
                prototype_gather_meta.append(torch.cat(prototype_gather))
                logger.info("Nums of prototypes of coarse clusters for meta train dataset {}: {}".format(
                    self.args.meta_datasets[task_idx], prototype_gather_meta[task_idx].size(0)))

            self.num_coarse_classes_meta = num_coarse_classes_meta
            self.prototype_gather_meta = prototype_gather_meta

        else:
            train_loader, _, _ = data_loader
            hc = cluster.AgglomerativeClustering(
                n_clusters=None, 
                linkage='average', 
                distance_threshold=threshold_dict[self.args.pretrained_model]
            )
            with torch.no_grad():
                for i, sample in enumerate(train_loader):
                    image = sample["image"].to(self.args.device)
                    rep = self.model.forward_features(image)
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


    def get_prompted_image(self, image, prototype_gather=None, prompter=None, prompter_gather=None):
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
                rep_batch = self.model.forward_features(image) # [N, emd_dim]
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


    def finetuning(self, test_data, prompter=None):
        """Evaluation for meta trained prompt.
        """
        train_loader, val_loader, test_loader = test_data
        if self.args.wo_da:
            prompter = deepcopy(prompter) if prompter else deepcopy(self.prompter)
            optimizer = torch.optim.SGD(
                prompter.parameters(),
                lr=1e+4,
                momentum=0.9,
                weight_decay=0
            )
        else:
            prompter_gather, prompter_params_gather = [], []
            for i in range(self.num_coarse_classes):
                prompter_gather.append(
                    deepcopy(prompter) if prompter else deepcopy(self.prompter)
                )
                prompter_params_gather.append(
                    {'params':prompter_gather[i].parameters()}
                )
            optimizer = torch.optim.SGD(
                prompter_params_gather,
                lr=1e+4,
                momentum=0.9,
                weight_decay=0
            )

        scheduler = cosine_lr(optimizer, 1e+4, len(train_loader)*10, len(train_loader)*50)
        num_classes = data_loader._dataset_class_num(self.args.test_dataset)
        BEST_ACC_VAL = -np.inf
        if self.args.wo_da:
            best_prompter = deepcopy(prompter)
        else:
            best_prompter_gather = deepcopy(prompter_gather)
        for epoch in range(20):
            # train
            for i, sample in enumerate(train_loader):
                # adjust learning rate
                global_step = len(train_loader) * epoch + i
                scheduler(global_step)
                image = sample["image"].to(self.args.device)
                label = sample["label"].to(self.args.device)
                prompted_image = self.get_prompted_image(image, prompter=prompter) \
                    if self.args.wo_da else self.get_prompted_image(image, self.prototype_gather, prompter_gather=prompter_gather)
                output = self.model.forward_features(prompted_image)
                logits = self.rep2logit(output, num_classes)
                loss = self.loss_function(logits, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if (i + 1) % 1 == 0:
                    logger.info("[Prompt Finetuning] Epoch: [{}/{}], Step: [{}/{}], Training loss: {}".format(
                        epoch, 49, i, len(train_loader)-1, loss.item()))
            # validate
            with torch.no_grad():
                num_total, correct = 0, 0
                for i, sample in enumerate(val_loader):
                    image = sample["image"].to(self.args.device)
                    label = sample["label"].to(self.args.device)
                    prompted_image = self.get_prompted_image(image, prompter=prompter) \
                        if self.args.wo_da else self.get_prompted_image(image, self.prototype_gather, prompter_gather=prompter_gather)
                    output = self.model.forward_features(prompted_image)
                    logits = self.rep2logit(output, num_classes)
                    pred = torch.argmax(logits, dim=-1)
                    correct += (pred == label).sum().item()
                    num_total += image.size(0)
                acc_val = float(correct / num_total)
                logger.info("[Prompt Finetuning] Epoch: {}, Val acc: {}".format(epoch, acc_val))
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
                    if self.args.wo_da else self.get_prompted_image(image, self.prototype_gather, prompter_gather=best_prompter_gather)
                output = self.model.forward_features(prompted_image)
                logits = self.rep2logit(output, num_classes)
                pred = torch.argmax(logits, dim=-1)
                correct += (pred == label).sum().item()
                num_total += image.size(0)
            acc_test = float(correct / num_total)
        return acc_test


