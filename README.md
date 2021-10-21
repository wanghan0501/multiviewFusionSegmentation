# Multi-view Fusion Segmentation for Brain Glioma on CT images

## Description
This repository is the implementation of the paper "[Multi-view Fusion Segmentation for Brain Glioma on CT images](https://link.springer.com/10.1007/s10489-021-02784-7)". 


## Contents
In addition to our proposed method, this repository also contains the implementation of other segmentation methods, including 2D CNN-based methods, and 3D CNN-based methods. 

The 2D CNN-baased methods include:

- DeepLabV3+
- DenseAspp
- DDCNN
- ResUNet

The 3D CNN-based methods include:

- VNet
- VoxResNet

## How tu use

The following description is about how to use our code.

### Train

Our code entry file is **segment_main.py**. 

The command line parameters of it include:

```
usage: segment_main.py [-h] [--seed SEED] [--use_cuda USE_CUDA] [--use_parallel USE_PARALLEL] [--gpu GPU] [--model {2D,2.5D,3D}] [--logdir LOGDIR]
                       [--train_sample_csv TRAIN_SAMPLE_CSV] [--eval_sample_csv EVAL_SAMPLE_CSV] [--weight WEIGHT] [--mod_root MOD_ROOT] [--config CONFIG]

PyTorch Radiology Segmentation

optional arguments:
  -h, --help            show this help message and exit
  --seed SEED           random seed for training. default=42
  --use_cuda USE_CUDA   whether use cuda. default: true
  --use_parallel USE_PARALLEL
                        whether use parallel. default: false
  --gpu GPU             use gpu device. default: all
  --model {2D,2.5D,3D}  which model used. default: 2D
  --logdir LOGDIR       which logdir used. default: None
  --train_sample_csv TRAIN_SAMPLE_CSV
                        train sample csv file used. default: None
  --eval_sample_csv EVAL_SAMPLE_CSV
                        eval sample csv file used. default: None
  --weight WEIGHT       criterion weight. default: None
  --mod_root MOD_ROOT   modification dir used. default: None
  --config CONFIG       configuration file. default: cfgs/seg2d/segment.yaml
```

### Test
Please use the notebook to test the trained model. We provide [an example](scripts_eval/(multi_fusion_segment)eval_for_test.ipynb) of how to evaluate your model.


### Configuration

We use the yaml files to configure some parameters. The yaml files can be found in [cfgs](cfgs/) folder.


## Dataset
Due to privacy restrictions, we only upload part of the test data used in the paper. If you are interested in our paper, please contact us.

## Cite
If you use our code, please cite:

```
@article{han2021multi,
  title={Multi-view fusion segmentation for brain glioma on CT images},
  author={Han Wang, Junjie Hu, Ying Song, Lei Zhang, Sen Bai, Zhang Yi},
  journal={Applied Intelligence},
  year={2021},
  doi={10.1007/s10489-021-02784-7}
}
```
