import tensorflow_datasets as tfds
# from tensorflow_datasets.core import download
data_dir = "/data-x/g12/huangqidong/VTAB/"  # TODO: setup the data_dir to put the the data to, the DATA.DATAPATH value in config

# download_config = download.DownloadConfig()
# download_config.download_mode = 'reuse_dataset_if_exists'

# caltech101
dataset_builder = tfds.builder("caltech101:3.0.1", data_dir=data_dir)
dataset_builder.download_and_prepare()
# dataset_builder, n = tfds.load("caltech101:3.0.1", data_dir="/data-x/g12/huangqidong/VTAB/")

# cifar100
dataset_builder = tfds.builder("cifar100:3.0.2", data_dir=data_dir)
dataset_builder.download_and_prepare()

# clevr
dataset_builder = tfds.builder("clevr:3.1.0", data_dir=data_dir)
dataset_builder.download_and_prepare()

# dmlab
dataset_builder = tfds.builder("dmlab:2.0.1", data_dir=data_dir)
dataset_builder.download_and_prepare()

# dsprites
dataset_builder = tfds.builder("dsprites:2.0.0", data_dir=data_dir)
dataset_builder.download_and_prepare()

# dtd
dataset_builder = tfds.builder("dtd:3.0.1", data_dir=data_dir)
dataset_builder.download_and_prepare()

# eurosat
subset="rgb"
dataset_name = "eurosat/{}:2.0.0".format(subset)
dataset_builder = tfds.builder(dataset_name, data_dir=data_dir)
dataset_builder.download_and_prepare()

# oxford_flowers102
dataset_builder = tfds.builder("oxford_flowers102:2.1.1", data_dir=data_dir)
dataset_builder.download_and_prepare()

# oxford_iiit_pet
dataset_builder = tfds.builder("oxford_iiit_pet:3.2.0", data_dir=data_dir)
dataset_builder.download_and_prepare()

# patch_camelyon
dataset_builder = tfds.builder("patch_camelyon:2.0.0", data_dir=data_dir)
dataset_builder.download_and_prepare()

# smallnorb
dataset_builder = tfds.builder("smallnorb:2.0.0", data_dir=data_dir)
dataset_builder.download_and_prepare()

# svhn
dataset_builder = tfds.builder("svhn_cropped:3.0.0", data_dir=data_dir)
dataset_builder.download_and_prepare()
# dataset_builder = tfds.load("svhn_cropped:3.0.0", data_dir="/data-x/g12/huangqidong/VTAB/")

# sun397 --> need cv2
# cannot load one image, similar to issue here: https://github.com/tensorflow/datasets/issues/2889
# "Image /t/track/outdoor/sun_aophkoiosslinihb.jpg could not be decoded by Tensorflow.""
# sol: modify the file: "/fsx/menglin/conda/envs/prompt_tf/lib/python3.7/site-packages/tensorflow_datasets/image_classification/sun.py" to ignore those images
dataset_builder = tfds.builder("sun397/tfds:4.*.*", data_dir=data_dir)
dataset_builder.download_and_prepare()

# kitti version is wrong from vtab repo, try 3.2.0 (https://github.com/google-research/task_adaptation/issues/18)
dataset_builder = tfds.builder("kitti:3.3.0", data_dir=data_dir)
dataset_builder.download_and_prepare()

# diabetic_retinopathy
"""
Download this dataset from Kaggle.
https://www.kaggle.com/c/diabetic-retinopathy-detection/data
After downloading, 
- unpack the test.zip file into <data_dir>/manual_dir/.
- unpack the sample.zip to sample/. 
- unpack the sampleSubmissions.csv and trainLabels.csv.

# ==== important! ====
# 1. make sure to check that there are 5 train.zip files instead of 4 (somehow if you chose to download all from kaggle, the train.zip.005 file is missing)
# 2. if unzip train.zip ran into issues, try to use jar xvf train.zip to handle huge zip file
cat test.zip.* > test.zip
cat train.zip.* > train.zip
"""

config_and_version = "btgraham-300" + ":3.0.0"
dataset_builder = tfds.builder("diabetic_retinopathy_detection/{}".format(config_and_version), data_dir=data_dir)
# dataset_builder = tfds.builder("{}".format(config_and_version), data_dir=data_dir)
dataset_builder.download_and_prepare()


# resisc45
"""
download/extract dataset artifacts manually: 
Dataset can be downloaded from OneDrive: https://1drv.ms/u/s!AmgKYzARBl5ca3HNaHIlzp_IXjs
After downloading the rar file, please extract it to the manual_dir.
"""

dataset_builder = tfds.builder("resisc45:3.0.0", data_dir=data_dir)
dataset_builder.download_and_prepare()
