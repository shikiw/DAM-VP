#!/usr/bin/env python3
from __future__ import print_function

import os
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
from meta_hf import *
from data_utils import loader as data_loader
from models import builder as model_builder
import utils.logging as logging
from launch import logging_train_setup



def load_meta_dataset(args):
    """Load datasets for meta training."""
    # load train
    minis = []
    for i in range(args.num_tasks):
        mini = data_loader.construct_train_loader(args, args.meta_datasets[i])
        minis.append(mini)
    # load test
    minis_test = [
        data_loader.construct_train_loader(args, args.test_dataset), 
        data_loader.construct_val_loader(args, args.test_dataset), 
        data_loader.construct_test_loader(args, args.test_dataset)
    ]
    return minis, minis_test


def main():
    """Meta training for initializing visual prompts."""
    assert args.num_tasks == len(args.meta_datasets)

    # load datasets for meta train or test
    minis, minis_test = load_meta_dataset(args)

    # load pretrained model
    model, cur_device = model_builder._construct_model(args)

    # initialize meta-learner
    metalearner = Meta(args, model).to(cur_device)

    # start training
    step_number = min([len(mini) for mini in minis])
    test_step_number = len(minis_test)
    BEST_LOSS = np.inf
    BEST_PROMPTER = metalearner.prompter
    BEST_TEST_ACC = -np.inf
    BEST_TEST_EPOCH = -1
    global_step = 0
    test_acc_list = []

    # inner functions
    def save_prompter(prompter, acc, epoch):
        file_path = "./checkpoints_" + str(args.pretrained_model) + "_wo_head/lr" + str(args.meta_lr) + \
            "_step" + str(args.update_step) + "_eps" + str(args.meta_step_size)
        if not os.path.exists(file_path):
            os.makedirs(file_path)
    
        file_name = args.prompt_method + "_Epoch" + str(epoch) + "_" + str(acc) + ".pth"
        save_model_path = os.path.join(file_path, file_name)
        state = {
            'state_dict': prompter.state_dict(),
            'epoch': epoch
        }
        torch.save(state, save_model_path)

    def load_model(model, acc):
        model_file_path = os.path.join('./checkpoints/', args.dataset)
        file_name = str(acc) + '_' + args.target_model + '.pth'
        model_checkpoint_path = os.path.join(model_file_path, file_name)
        assert os.path.exists(model_checkpoint_path)
        model.net.load_state_dict(torch.load(model_checkpoint_path))        
        return model

    # use diverisity aware
    if not args.wo_da:
        # coarse clustering
        metalearner.coarse_clustering(minis, "meta_training")
        metalearner.coarse_clustering(minis_test, "task_adapting")

    # main training loop
    for epoch in range(args.epochs):
        # get data iterator
        minis_iter = []
        for i in range(len(minis)):
            minis_iter.append(iter(minis[i]))

        # evaluation
        if epoch == 0:
            # baseline performance
            accs = metalearner.finetuning(minis_test)
            test_acc_list.append(accs)
            logger.info('[Prompt Finetuning] Testing acc on {}: {}'.format(args.test_dataset, accs))

        # update loop for each meta batch
        for step in range(step_number):
            try:
                batch_data = []
                for each in minis_iter:
                    batch_data.append(each.next())
            except:
                break

            global_step += 1
            metalearner.lr_scheduler(global_step, args.epochs*step_number, 0)

            accs = metalearner(batch_data)
            if (step + 1) % 1 == 0:
                logger.info("[Meta Training] Epoch: [{}/{}], Step: [{}/{}], Training loss: {}".format(
                    epoch, args.epochs-1, step, step_number-1, accs[-1]))
            if accs[-1] < BEST_LOSS:
                BEST_LOSS = accs[-1]
                BEST_PROMPTER = deepcopy(metalearner.prompter)
                logger.info("[Meta Training] Epoch: [{}/{}], Step: [{}/{}], Current best loss: {}".format(
                    epoch, args.epochs-1, step, step_number-1, BEST_LOSS))
                save_prompter(BEST_PROMPTER, BEST_LOSS, epoch)

         # evaluation
        if (epoch + 1) % 1 == 0:
            accs = metalearner.finetuning(minis_test, BEST_PROMPTER)
            test_acc_list.append(accs)
            logger.info('[Prompt Finetuning] Testing acc on {}: {}'.format(args.test_dataset, accs))
            # logger.info(test_acc_list)
            if accs >= BEST_TEST_ACC:
                BEST_TEST_ACC = accs
                BEST_TEST_EPOCH = epoch
                save_prompter(BEST_PROMPTER, accs, BEST_TEST_EPOCH)
            logger.info("[Prompt Finetuning] Current Best Evaluation is Epoch: {}, Acc: {}".format(BEST_TEST_EPOCH, BEST_TEST_ACC))
        logger.info(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

    accs = metalearner.finetuning(minis_test)
    logger.info('[Prompt Finetuning] Testing acc on {} after the last epoch: {}'.format(args.test_dataset, accs))





if __name__ == '__main__':
    args = Arguments(stage='meta_training').parser().parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # setup training env including loggers
    logging_train_setup(args)
    logger = logging.get_logger("dam-vp")

    # basic configuration
    set_seed(args.seed)
    logger.info(f"Using random seed: {args.seed}")

    # main loop
    main()