#!/bin/bash

wget https://s3.amazonaws.com/fast-ai-imageclas/imagewoof2.tgz

ls
tar -xzf imagewoof2.tgz
mkdir ImageNet
mv imagewoof2/ ImageNet/
mv ImageNet/imagewoof2/ ImageNet/train
mkdir ImageNet/test