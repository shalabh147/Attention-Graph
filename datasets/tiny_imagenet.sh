#!/bin/bash

wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
ls
unzip tiny-imagenet-200.zip
rm -r tiny-imagenet-200/test
mkdir tiny-imagenet-200/test

python3 tiny_imagenet_helper.py