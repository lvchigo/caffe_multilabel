export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/xiaotian/caffe/build/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:../lib

./demo $1 $2 1 1 $3 paths.csv paths.txt features.txt
