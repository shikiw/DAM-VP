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
For more tips about how to download VTAB-1k, please refer to [VTAB_SETUP.md](https://github.com/KMnP/vpt/blob/main/VTAB_SETUP.md).

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

## Pre-trained Model Preparation
The used pre-trained vision models are detailed in Table 8 of our paper. Their checkpoints can be downloaded here:

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Backbone</th>
<th valign="bottom">Pre-trained Objective</th>
<th valign="bottom">Pre-trained Dataset</th>
<th valign="bottom">Download</th>
<th valign="bottom">md5sum</th>
<!-- TABLE BODY -->
<tr><td align="left">ViT-B/16</td>
<td align="center">Supervised</td>
<td align="center">ImageNet-1k</td>
<td align="center"><a href="xxx">Download</a></td>
<td align="center"><tt>-</tt></td>
</tr>
<tr><td align="left">ViT-B/16</td>
<td align="center">Supervised</td>
<td align="center">ImageNet-22k</td>
<td align="center"><a href="xxx">Download</a></td>
<td align="center"><tt>-</tt></td>
</tr>
<tr><td align="left">ViT-B/16</td>
<td align="center">CLIP</td>
<td align="center">400M Web Data</td>
<td align="center"><a href="https://openai.com/research/clip">Download</a></td>
<td align="center"><tt>-</tt></td>
</tr>
<tr><td align="left">Swin-B</td>
<td align="center">Supervised</td>
<td align="center">ImageNet-22k</td>
<td align="center"><a href="https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22k.pth">Download</a></td>
<td align="center"><tt>bf9cc1</tt></td>
</tr>
<tr><td align="left">ViT-B/16</td>
<td align="center">MoCo v3</td>
<td align="center">ImageNet-1k</td>
<td align="center"><a href="xxx">Download</a></td>
<td align="center"><tt>-</tt></td>
</tr>
<tr><td align="left">ResNet-50</td>
<td align="center">Supervised</td>
<td align="center">ImageNet-1k</td>
<td align="center"><a href="https://pytorch.org/vision/stable/models.html">Download</a></td>
<td align="center"><tt>-</tt></td>
</tr>
</tbody></table>


## Meta Prompt Initialization
The trained meta prompts are available at [here](xxx), you can directly use these checkpoints without meta training.
To implement the meta training of visual prompts, you can refer to the following instructions.
* For head-freezing/missing scenario, please run the command:
```
cd meta-training/
# if training on vit-b-1k
python main_hf.py --base_dir /your/path/to/dataset/ --pretrained_model vit-b-1k --meta_lr 0.5 --update_lr 0.5 --update_step 4 --meta_step_size 0.5 --test_dataset oxford-flowers
# if training on clip-vit-b
python main_clip.py --base_dir /your/path/to/dataset/  --pretrained_model clip-vit-b --meta_lr 1.0 --update_lr 1.0 --update_step 4 --meta_step_size 0.5
```
* For head-tuning scenario, please run the command:
```
cd meta-training/
# if training on vit-b-22k
python main_ht.py --base_dir /your/path/to/dataset/ --pretrained_model vit-b-22k --meta_lr 1.0 --update_lr 1.0 --update_step 4 --meta_step_size 0.5 --weight_decay 1e-4  --test_dataset oxford-flowers
# if training on swin-b-22k
python main_ht.py --base_dir /your/path/to/dataset/ --pretrained_model swin-b-22k --meta_lr 0.5 --update_lr 0.5 --update_step 4 --meta_step_size 0.5 --weight_decay 1e-4
# if training on moco-v3-b-1k
python main_ht.py --base_dir /your/path/to/dataset/ --pretrained_model moco-v3-b-1k --meta_lr 0.5 --update_lr 0.5 --update_step 4 --meta_step_size 0.5 --weight_decay 1e-4
# if training on resnet50-1k
python main_ht.py --base_dir /your/path/to/dataset/ --pretrained_model resnet50-1k --meta_lr 0.5 --update_lr 0.5 --update_step 4 --meta_step_size 0.5 --weight_decay 1e-4
```

## Diversity-Aware Prompting
```
```

## Citation
If you find this work useful for your research, please cite [our paper](https://arxiv.org/abs/2303.08138):
```
@inproceedings{huang2023damvp,
  title={Diversity-Aware Meta Visual Prompting},
  author={Qidong Huang and Xiaoyi Dong and Dongdong Chen and Weiming Zhang and Feifei Wang and Gang Hua and Nenghai Yu},
  journal={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2023}
}
```

## License
The code is released under MIT License (see LICENSE file for details).

