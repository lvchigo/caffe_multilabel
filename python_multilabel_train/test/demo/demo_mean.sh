export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/xiaotian/caffe/build/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:../lib

./demo_mean /home/xiaotian/models/resNet/ResNet-101-deploy.prototxt /home/xiaotian/models/resNet/ResNet_mean.binaryproto /home/xiaotian/models/resNet/ResNet-101-model.caffemodel 1 1 pool5 paths.csv paths.txt features.txt
