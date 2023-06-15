# Locally Linear Embedding Unbiased Domain Randomization Networks for Cross-Scene Hyperspectral Image Classification

<p align='center'>
  <img src='abstract.png' width="800px">
</p>

## Abstract

Unseen target domain is often inevitable, especially for hyperspectral classification applications processed offline on edge devices. One feasible idea in domain generalization is to calculate potential values for performing domain expansion, and optimize discriminator to learn domain-invariant representation. First, for domain extension, this paper proposes a unbiased style generation network, which consists of linear mapping and 3D convolution to implicitly learn space-spectrum local feature and reverse reconstruction. In particular, intra-class supervision contrastive learning is employed to prevent redundant extension. For the discriminator, a 2D convolution backbone with classification and feature projection is designed. Considering low-density separation assumptions, an inter-class supervision comparison penalty item is embedded in optimization step. Extensive experiments on two public HSI datasets demonstrate the superiority of the proposed method when compared with state-of-the-art techniques.
## Requirements

CUDA Version: 11.7

torch: 2.0.0

Python: 3.10

## Dataset

The dataset directory should look like this:

```bash
datasets
├── Houston
│   ├── Houston13.mat
│   ├── Houston13_7gt.mat
│   ├── Houston18.mat
│   └── Houston18_7gt.mat
└── Pavia
    ├── paviaC.mat
    └── paviaC_7gt.mat
    ├── paviaU.mat
    └── paviaU_7gt.mat

```

## Usage

1.You can download [Houston &amp; Pavia](https://drive.google.com/drive/folders/1No-DNDT9P1HKsM9QKKJJzat8A1ZhVmmz?usp=drive_link) dataset here.

2.You can change the `source_name` and `target_name` in train.py to set different transfer tasks.

3.Run the following command:

Houston dataset:
```
python train.py --data_path ./datasets/Houston/ --source_name Houston13 --target_name Houston18 --re_ratio 5 --training_sample_ratio 0.8 --dim1 128 --dim2 8 --lambda_1 1.0 --lambda_2 1.0
```
Pavia dataset:
```
python train.py --data_path ./datasets/Pavia/ --source_name paviaU --target_name paviaC --re_ratio 1 --training_sample_ratio 0.8 --dim1 8 --dim2 16 --lambda_1 1.0 --lambda_2 1.0
```

## Note

- The variable names of data and gt in .mat file are set as `ori_data` and `map`.
- For Pavia dataset and Houston dataset, args.re_ratio is set to 1 and 5, respectively
