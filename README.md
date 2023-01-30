# SENet-pytorch
Implementing **SENet** using the pytorch framework

## requirement
* python3.8
* pytorch1.6
* torchvision
* tensorboard
* tqdm

## dataset
The data set is automatically downloaded to the data folder, if the speed is too slow,You can try the following link
https://share.weiyun.com/56FKfYz
password：nwdmtc

## train
```
python train.py
```
If you want to train a model without SE block
```
python train.py --baseline
```
Trainers can change parameters in train.py according to their needs.

## log
The log file will be saved in the logs folder
```
 tensorboard --logdir=logs
```
You can view the training process and results

## some result
|                  | ResNet20       | SE-ResNet20 (reduction 4 or 8)    |
|:-------------    | :------------- | :------------- |
|max. test accuracy|  92%           | 93%            |
