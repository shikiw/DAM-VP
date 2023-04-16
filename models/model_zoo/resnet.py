#!/usr/bin/env python3
"""Deep Residual Network."""
import os
import sys
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, '../'))

from backbones import *


def ResNet50_1K(args):
    """Construct resnet-50 pretrained on ImageNet-1K."""
    model = resnet50(pretrained=True)
    return model


def ResNet101_1K(args):
    """Construct resnet-101 pretrained by ImageNet-1K."""
    model = resnet101(pretrained=True)
    return model
