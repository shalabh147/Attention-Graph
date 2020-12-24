#!/bin/bash

#!rm -r ImageNet
wget https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz
ls
tar -xzf imagenette2.tgz
mkdir ImageNet
mv imagenette2/ ImageNet/
mv ImageNet/imagenette2/ ImageNet/train
mkdir ImageNet/test