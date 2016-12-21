#!/usr/bin/env sh
# Compute the mean image from the imagenet training leveldb
# N.B. this is available in data/ilsvrc12

/home/xiaogao/caffe-master/build/tools/compute_image_mean lmdb_train\
  imagenet_mean.binaryproto

echo "Done."
