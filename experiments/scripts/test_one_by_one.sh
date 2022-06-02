#!/bin/bash


GPU_ID=$1
START_EPOCH=$2
END_EPOCH=$3
FILE=$4


for ((i=$START_EPOCH; i<=END_EPOCH; i+=5000))

do

bash /home/yjy123/github/original/pytorch_MELM-master/experiments/scripts/test_faster_rcnn_plus_iter.sh $GPU_ID pascal_voc vgg16 $i $FILE

done