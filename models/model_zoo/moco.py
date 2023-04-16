#!/usr/bin/env python3
"""Momentum Contrast for Unsupervised Visual Representation Learning."""
import os
import sys
import torch

from timm.models import create_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, '../'))

import backbones


def MoCo_v3_B_1K(args):
    """Construct moco_v3_b pretrained on ImageNet-1K."""
    model = create_model(
        'jx_moco_v3_base_patch16_224',
        pretrained=False,
        num_classes=1000,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
    )
    return _load_checkpoint(args, model)



def _load_checkpoint(args, model):
    """Load the checkpoint into the given model."""
    path = os.path.join(ROOT_DIR, "../checkpoints/moco_base_p16_224_in1k.pth")
    checkpoint = torch.load(path, map_location="cpu")

    if "module" in checkpoint:
        checkpoint = checkpoint["module"]
    # for key in list(checkpoint.keys()):
    #     if key in ["pre_logits.fc.bias", "pre_logits.fc.weight"]: # ["head.bias", "head.weight"]:
    #         del checkpoint[key]

    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    elif "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)

    return model