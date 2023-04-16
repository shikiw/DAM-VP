#!/usr/bin/env python3
"""Contrastive Language-Image Pre-Training."""
import os
import sys
import torch

import clip

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, '../'))

import backbones


def CLIP_ResNet50(args):
    """Construct resnet-50 pretrained by CLIP."""
    model, _ = clip.load('RN50', args.device, jit=False)
    return model


def CLIP_ViT_B(args):
    """Construct vit_b/16 pretrained by CLIP."""
    model, _ = clip.load('ViT-B/16', args.device, jit=False)
    return model
