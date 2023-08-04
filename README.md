# **HGR / GCN**

This repository contains the source code for our paper:

**Paper Title**

![hippo](images/approach.png)

## **Updates**
- ...

## **Installation**
Create and activate conda environment:
```
conda create -n GCN_env python=3.10
conda activate GCN_env
```

Install all dependencies:
```
pip install -r requirements.txt
```

### Demos

## Dataset

By default, the training datasets are structured as follows for each sequence (see example in `./datasets/`):

```
## frames 1
x1, y1, z1
x2, y2, z2
...
...
x21, y21, z21

## frame 2
x1, y1, z1
x2, y2, z2
...
...
x21, y21, z21

## frame t
...

## frame T
x1, y1, z1
x2, y2, z2
...
...
x21, y21, z21

```


## Training

```
python train.py --params
```


## Citation
If you find this repo useful, please consider citing our paper

```ref```
