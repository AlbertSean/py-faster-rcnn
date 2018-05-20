#!/bin/bash

mkdir -p ./data
mkdir -p ./output

cd lib/utils
python setup_linux.py build_ext --inplace
cd ../datasets/pycocotools
python setup_linux.py build_ext --inplace
cd ../../nms
python setup_linux.py build_ext --inplace
cd ../..
