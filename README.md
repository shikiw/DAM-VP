# Diversity-Aware Meta Visual Prompting (CVPR 2023)
This repository provides the official PyTorch implementation of the following conference paper: 
> [**Diversity-Aware Meta Visual Prompting (CVPR 2023)**](https://arxiv.org/abs/2303.08138) <br>
> [Qidong Huang](https://shikiw.github.io/)<sup>1</sup>, 
> [Xiaoyi Dong](https://scholar.google.com/citations?user=FscToE0AAAAJ&hl=en)<sup>1</sup>, 
> [Dongdong Chen](https://www.dongdongchen.bid/)<sup>2</sup>, 
> [Weiming Zhang](http://staff.ustc.edu.cn/~zhangwm/index.html)<sup>1</sup>, 
> [Feifei Wang](http://home.ustc.edu.cn/~wangfeifei/)<sup>1</sup>, 
> [Gang Hua](https://www.ganghua.org/)<sup>3</sup>, 
> [Nenghai Yu](https://scholar.google.com/citations?user=7620QAMAAAAJ&hl=en)<sup>1</sup> <br>
> <sup>1</sup>University of Science and Technology of China, <sup>2</sup>Microsoft Cloud AI, <sup>3</sup>Wormpex AI Research <br>
>

## Environment Setup
This code is tested with Python3.8, Pytorch = 1.11 and CUDA = 11.3, requiring the following dependencies:

* timm = 0.4.9
* lpips = 0.1.4
* opencv-python = 4.6.0.66

To setup a conda environment, please use the following instructions:
```
conda env create -f environment.yaml
conda activate dam_vp
```

## Dataset Preparation
The Fine-Grained Visual Classification (FGVC) datasets can be downloaded in [VPT repo](https://github.com/KMnP/vpt). The Fru92 and Veg200 datasets can be downloaded at [VegFru](https://github.com/ustc-vim/vegfru). Other datasets are all avaliable at torchvision. 
* (Optional) To prepare the datasets of [Visual Task Adaptation Benchmark (VTAB)](https://google-research.github.io/task_adaptation/) benchmark, you can install the tensorflow package as in [VPT repo](https://github.com/KMnP/vpt) and run the command below:
```
python data_utils/vtab_prep.py
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
<td align="center"><a href="https://drive.google.com/file/d/1_cunej-ZSB58ngtOW62mh0GxOFoQvnjY/view?usp=sharing">Download</a></td>
<td align="center"><tt>-</tt></td>
</tr>
<tr><td align="left">ViT-B/16</td>
<td align="center">Supervised</td>
<td align="center">ImageNet-22k</td>
<td align="center"><a href="https://drive.google.com/file/d/1zvIqdml4KVArPuWspoHKU7a6e0uAunF8/view?usp=sharing">Download</a></td>
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
<td align="center"><a href="https://drive.google.com/file/d/1w_7CVKKlRq_VT-M6-aYFu1UlrjMxgXGA/view?usp=sharing">Download</a></td>
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
The trained meta prompts are available at [here](https://drive.google.com/drive/folders/1X0ZgnQlZw57iqSxORS_n4A8JbxQVfd3q?usp=sharing), you can directly download these checkpoints and store them at ```./meta-training/checkpoints/```.
Also, you can implement the meta training of visual prompts by yourself. The following instructions will be helpful.
* For head-freezing/missing scenario, please run the command:
```
cd meta-training/
# if prompting on vit-b-1k
python main_hf.py --base_dir /your/path/to/dataset/ --pretrained_model vit-b-1k --meta_lr 0.5 --update_lr 0.5 --update_step 4 --meta_step_size 0.5 --test_dataset oxford-flowers
# if prompting on clip-vit-b
python main_clip.py --base_dir /your/path/to/dataset/  --pretrained_model clip-vit-b --meta_lr 1.0 --update_lr 1.0 --update_step 4 --meta_step_size 0.5
```
* For head-tuning scenario, please run the command:
```
cd meta-training/
# if prompting on vit-b-22k
python main_ht.py --base_dir /your/path/to/dataset/ --pretrained_model vit-b-22k --meta_lr 1.0 --update_lr 1.0 --update_step 4 --meta_step_size 0.5 --weight_decay 1e-4  --test_dataset oxford-flowers
# if prompting on swin-b-22k
python main_ht.py --base_dir /your/path/to/dataset/ --pretrained_model swin-b-22k --meta_lr 0.5 --update_lr 0.5 --update_step 4 --meta_step_size 0.5 --weight_decay 1e-4
# if prompting on moco-v3-b-1k
python main_ht.py --base_dir /your/path/to/dataset/ --pretrained_model moco-v3-b-1k --meta_lr 0.5 --update_lr 0.5 --update_step 4 --meta_step_size 0.5 --weight_decay 1e-4
# if prompting on resnet50-1k
python main_ht.py --base_dir /your/path/to/dataset/ --pretrained_model resnet50-1k --meta_lr 0.5 --update_lr 0.5 --update_step 4 --meta_step_size 0.5 --weight_decay 1e-4
```

## Diversity-Aware Prompting
With the meta trained visual prompt, we can adapt pretrained vision models to unseen vision datasets. The hyper-parameter configurations can be found in Table 13 and Table 14 of our paper. 
* For head-freezing/missing scenario, please run the command:
```
cd task_adapting/
# if prompting on vit-b-1k
python main.py --base_dir /your/path/to/dataset/ --pretrained_model vit-b-1k --adapt_method prompt_wo_head --test_dataset /select/one/dataset/ --epochs 50 --lr /learning/rate/ --weight_decay /weight/decay/rate/ --checkpoint_dir ../meta-training/checkpoints/vit-b-1k-wo-head.pth
# if prompting on clip-vit-b
python main_clip.py --base_dir /your/path/to/dataset/ --pretrained_model clip-vit-b --adapt_method prompt_wo_head --test_dataset /select/one/dataset/ --epochs 50 --lr /learning/rate/ --weight_decay /weight/decay/rate/ --checkpoint_dir ../meta-training/checkpoints/clip-vit-b-wo-head.pth
```
* For head-tuning scenario, please run the command:
```
cd task_adapting/
# if prompting on vit-b-22k
python main.py --base_dir /your/path/to/dataset/ --pretrained_model vit-b-22k --adapt_method ours_with_head --test_dataset /select/one/dataset/ --epochs 50 --lr /learning/rate/ --weight_decay /weight/decay/rate/ --checkpoint_dir ../meta-training/checkpoints/vit-b-22k-w-head.pth
# if prompting on swin-b-22k
python main.py --base_dir /your/path/to/dataset/ --pretrained_model swin-b-22k --adapt_method ours_with_head --test_dataset /select/one/dataset/ --epochs 50 --lr /learning/rate/ --weight_decay /weight/decay/rate/ --checkpoint_dir ../meta-training/checkpoints/swin-b-22k-w-head.pth
# if prompting on moco-v3-b-1k
python main.py --base_dir /your/path/to/dataset/ --pretrained_model moco-v3-b-1k --adapt_method ours_with_head --test_dataset /select/one/dataset/ --epochs 50 --lr /learning/rate/ --weight_decay /weight/decay/rate/ --checkpoint_dir ../meta-training/checkpoints/moco-v3-b-1k-w-head.pth
# if prompting on resnet50-1k
python main.py --base_dir /your/path/to/dataset/ --pretrained_model resnet50-1k --adapt_method ours_with_head --test_dataset /select/one/dataset/ --epochs 50 --lr /learning/rate/ --weight_decay /weight/decay/rate/ --checkpoint_dir ../meta-training/checkpoints/resnet50-1k-w-head.pth
```

## Acknowledgement
This repo is partially based on [VP](https://github.com/hjbahng/visual_prompting) and [VPT](https://github.com/KMnP/vpt). Thanks for their impressive works!

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

