
#Get Train DB
#bash create_imagenet.sh

#Get Mean
#bash make_imagenet_mean.sh

#Train
#/home/xiaogao/caffe-master/build/tools/caffe train --solver=solver.prototxt

#ReTrain
#/home/xiaogao/caffe-master/build/tools/caffe train --solver=solver.prototxt --snapshot=caffenet_train_iter_19000.solverstate

#Fineturn
/home/xiaogao/caffe-master/build/tools/caffe train --solver=solver.prototxt --weights=bvlc_reference_caffenet.caffemodel

#Get Time
#/home/xiaogao/caffe-master/build/tools/caffe time --model=train_val.prototxt



