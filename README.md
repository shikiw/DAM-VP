# Diversity-Aware Meta Visual Prompting (CVPR 2023)
This repository provides the official PyTorch implementation of the following conference paper: 
> [**Diversity-Aware Meta Visual Prompting (CVPR 2023)**](https://arxiv.org/abs/2303.08138) <br>
> [Qidong Huang](http://home.ustc.edu.cn/~hqd0037/)<sup>1</sup>, 
> [Xiaoyi Dong](https://scholar.google.com/citations?user=FscToE0AAAAJ&hl=en)<sup>1</sup>, 
> [Dongdong Chen](https://www.dongdongchen.bid/)<sup>2</sup>, 
> [Weiming Zhang](http://staff.ustc.edu.cn/~zhangwm/index.html)<sup>1</sup>, 
> [Feifei Wang](http://home.ustc.edu.cn/~wangfeifei/)<sup>1</sup>, 
> [Gang Hua](https://www.ganghua.org/)<sup>3</sup>, 
> [Nenghai Yu](https://scholar.google.com/citations?user=7620QAMAAAAJ&hl=en)<sup>1</sup> <br>
> <sup>1</sup>University of Science and Technology of China, <sup>2</sup>Microsoft Cloud AI, <sup>3</sup>Wormpex AI Research <br>
>

## Environment Setup
This code is tested with Python3.8, Pytorch >= 1.11 and CUDA >= 11.3, requiring the following dependencies:

* timm = 0.4.9
* lpips = 0.1.4
* opencv-python = 4.6.0.66

To setup a conda environment, please use the following instructions:
```
conda env create -f environment.yml
conda activate dam_vp
```

## Dataset Preparation
The Fine-Grained Visual Classification (FGVC) datasets can be downloaded in [VPT repo](https://github.com/KMnP/vpt). The Fru92 and Veg200 datasets can be downloaded at [VegFru](https://github.com/ustc-vim/vegfru). Other datasets are all avaliable at torchvision. 
* (Optional) To prepare the datasets of [Visual Task Adaptation Benchmark (VTAB)](https://google-research.github.io/task_adaptation/) benchmark, you can install the tensorflow package as in [VPT repo](https://github.com/KMnP/vpt) and run the command below:
```
python vtab_prep.py
```
If more download tips about VTAB-1k are expected, you can refer to [VTAB_SETUP.md](https://github.com/KMnP/vpt/blob/main/VTAB_SETUP.md).

The overall directory structure should be:
```
│DAM-VP/
├──data/
│   ├──FGVC/
│   │   ├──CUB_200_2011/
│   │   ├──OxfordFlower/
│   │   ├──Stanford-cars/
│   │   ├──Stanford-dogs/
│   │   ├──nabirds/
│   ├──VTAB/
│   │   ├──.......
│   ├──finegrained_dataset/
│   │   ├──vegfru-dataset/
│   ├──torchvision_dataset/
│   │   ├──.......
├──.......
```
