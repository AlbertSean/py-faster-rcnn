#!/bin/bash
# Usage:
# ./experiments/scripts/faster_rcnn_end2end.sh GPU NET DATASET [options args to {train,test}_net.py]
# DATASET is either pascal_voc or coco.
#
# Example:
# ./experiments/scripts/faster_rcnn_end2end.sh 0 VGG_CNN_M_1024 pascal_voc \
#   --set EXP_DIR foobar RNG_SEED 42 TRAIN.SCALES "[400, 500, 600, 700]"

set -x
set -e

export PYTHONUNBUFFERED="True"

GPU_ID=$1
NET=$2
#ResNet101_BN_SCALE_Merged
#ResNet101_BN_SCALE_Merged_OHEM

NET_lc=${NET,,}
DATASET=$3

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

case $DATASET in
  GTSDB)
    TRAIN_IMDB="GTSDB_train"
    TEST_IMDB="GTSDB_test"
    PT_DIR="GTSDB"
    ITERS=50000
    ;;
  traffic_sign)
    TRAIN_IMDB="traffic_sign_train"
    TEST_IMDB="traffic_sign_val"
    PT_DIR="traffic_sign"
    ITERS=1000
    ;;
  pascal_voc)
    #TRAIN_IMDB="voc_2007_trainval"
    TRAIN_IMDB="VOC2007_trainval"
    #TEST_IMDB="voc_2007_test"
    TEST_IMDB="VOC2007_test"
    PT_DIR="pascal_voc"
    ITERS=1000
    ;;
  coco)
    # This is a very long and slow training schedule
    # You can probably use fewer iterations and reduce the
    # time to the LR drop (set in the solver to 350,000 iterations).
    TRAIN_IMDB="coco_2014_train"
    TEST_IMDB="coco_2014_minival"
    PT_DIR="coco"
    ITERS=490000
    ;;
  *)
    echo "No dataset given"
    exit
    ;;
esac

case $NET_lc in
    resnet50)
    PRE_MODEL=$HOME/data/pretrained_caffemodel/imagenet_models/ResNet-50-model.caffemodel
    NET_FULL='ResNet50_BN_SCALE_Merged'
    ;;
    resnet50_ohem)
    PRE_MODEL=$HOME/data/pretrained_caffemodel/imagenet_models/ResNet-50-model.caffemodel
    NET_FULL='ResNet50_BN_SCALE_Merged_OHEM'
    ;;
    resnet101)
    PRE_MODEL=$HOME/data/pretrained_caffemodel/imagenet_models/ResNet101_BN_SCALE_Merged.caffemodel
    NET_FULL='ResNet101_BN_SCALE_Merged'
    ;;
  resnet101_ohem)
    PRE_MODEL=$HOME/data/pretrained_caffemodel/imagenet_models/ResNet101_BN_SCALE_Merged.caffemodel
    NET_FULL='ResNet101_BN_SCALE_Merged_OHEM'
    ;;
   *)
    echo "Unknown resnet type"
    exit
    ;;
esac



LOG="experiments/logs/faster_rcnn_end2end_${NET}_${EXTRA_ARGS_SLUG}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

RESNET101=resnet101_faster_rcnn_bn_scale_merged_end2end_iter_70000.caffemodel

time ./tools/train_net.py --gpu ${GPU_ID} \
  --solver models/${PT_DIR}/${NET_FULL}/faster_rcnn_end2end/solver.prototxt \
  --weights ${PRE_MODEL} \
  --imdb ${TRAIN_IMDB} \
  --iters ${ITERS} \
  --cfg experiments/cfgs/faster_rcnn_end2end.yml \
  ${EXTRA_ARGS}

set +x
NET_FINAL=`grep -B 1 "done solving" ${LOG} | grep "Wrote snapshot" | awk '{print $4}'`
#NET_FINAL=output/faster_rcnn_end2end/traffic_sign_train/vgg_cnn_m_1024_faster_rcnn_iter_1000.caffemodel
#NET_FINAL=output/faster_rcnn_end2end/GTSDB_train/vgg_cnn_m_1024_faster_rcnn_iter_50000.caffemodeldd
#NET_FINAL=output/faster_rcnn_end2end/VOC2007_trainval/resnet50_faster_rcnn_bn_scale_merged_end2end_iter_1000.caffemodel
set -x

time ./tools/test_net.py --gpu ${GPU_ID} \
  --def models/${PT_DIR}/${NET_FULL}/faster_rcnn_end2end/test.prototxt \
  --net ${NET_FINAL} \
  --imdb ${TEST_IMDB} \
  --cfg experiments/cfgs/faster_rcnn_end2end.yml \
  ${EXTRA_ARGS}
