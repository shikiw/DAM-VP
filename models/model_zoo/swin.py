#!/usr/bin/env python3
"""Vision Transformer."""
import os
import sys
import torch

from timm.models import create_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, '../'))

import backbones



def Swin_B_1K(args):
    """Construct swin_base_patch4_window7_224 pretrained on ImageNet-21K, finetuned on ImageNet-1K."""
    model = create_model(
        'jx_swin_base_patch4_window7_224',
        pretrained=False,
        num_classes=1000,
    )
    return _load_checkpoint(args, model)


def Swin_B_22K(args):
    """Construct swin_base_patch4_window7_224 pretrained on ImageNet-22K."""
    model = create_model(
        'jx_swin_base_patch4_window7_224_in22k',
        pretrained=False,
        num_classes=21841,
    )
    return _load_checkpoint(args, model)


def _load_checkpoint(args, model):
    """Load the checkpoint into the given model."""
    if args.pretrained_model == "swin-b-1k":
        path = os.path.join(ROOT_DIR, "../checkpoints/swin_base_patch4_window7_224.pth")
    elif args.pretrained_model == "swin-b-22k":
        path = os.path.join(ROOT_DIR, "../checkpoints/swin_base_patch4_window7_224_22k.pth")
    else:
        raise NotImplementedError
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