#!/bin/bash

arch=resenet # can be one of these: resnet, resenet or seresnet
depth=20 # any valid resnet depths can be given
dataset=cifar10 # can be one of these: cifar10 or cifar100
gpuid=0 # in a multi-GPU environment, if you want training on a particular GPU, then give its id.
savename=resenet20 # any valid name for saving the model

#To train the model run the following command. As far as possible leave the hyperparameters

python cifar.py -a $arch --depth $depth --epochs 164 --schedule 81 122 --gamma 0.1 --wd 1e-4 --checkpoint checkpoints/$dataset/resenet/ --gpu-id $gpuid --saveModel $savename --rounds 1  --dataset $dataset
