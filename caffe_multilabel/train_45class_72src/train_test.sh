
#Get Train DB
#bash create_imagenet.sh

#Get Mean
#bash make_imagenet_mean.sh

#Train
#/home/xiaogao/caffe-master/build/tools/caffe train --solver=solver.prototxt --gpu 1

#ReTrain
#/home/xiaogao/caffe-master/build/tools/caffe train --solver=solver.prototxt --snapshot=caffenet_train_iter_5000.solverstate --gpu 1

#Fineturn
/home/xiaogao/caffe-master/build/tools/caffe train --solver=solver.prototxt --weights=bvlc_reference_caffenet.caffemodel --gpu 1

#Get Time
#/home/xiaogao/caffe-master/build/tools/caffe time --model=train_val.prototxt



