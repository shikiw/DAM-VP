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

## Setup
This code is tested with Python3.8, Pytorch == 1.11 and CUDA == 11.3, requiring the following dependencies:

* timm == 0.4.9
* lpips == 0.1.4
* opencv-python == 4.6.0.66

To setup a conda environment, please follow the bash instructions as below:
```
conda env create -f environment.yml
conda activate dam_vp
```
