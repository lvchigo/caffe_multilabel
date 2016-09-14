import sys 
import os

import numpy as np
import os.path as osp
import matplotlib.pyplot as plt

#% matplotlib inline
#plt.rcParams['figure.figsize'] = (6, 6)

from copy import copy

caffe_root = '/home/chigo/caffe_tts/caffe-master-20160913/'
sys.path.append(caffe_root + 'python')
import caffe

from caffe import layers as L, params as P
sys.path.append("/home/chigo/caffe_tts/caffe-master-20160913/examples/pycaffe/layers") # the datalayers we will use are in this directory.
sys.path.append("/home/chigo/caffe_tts/caffe-master-20160913/examples/pycaffe") # the tools file is in this folder

import tools #this contains some tools that we need

# set data root directory, e.g:
#pascal_root = osp.join(caffe_root, 'data/pascal/VOC2007/') #voc2007
pascal_root = osp.join(caffe_root, 'data/pascal/itugo_doudou/') #itugo_doudou

#LABEL_NUM = 20; #label
LABEL_NUM = 473; #label

# these are the PASCAL classes, we'll need them later.
#use classes just for plot image
#classes = np.asarray(['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'])


# helper function for common structures
def conv_relu(bottom, ks, nout, stride=1, pad=0, group=1):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                                num_output=nout, pad=pad, group=group)
    return conv, L.ReLU(conv, in_place=True)

# another helper function
def fc_relu(bottom, nout):
    fc = L.InnerProduct(bottom, num_output=nout)
    return fc, L.ReLU(fc, in_place=True)

# yet another helper function
def max_pool(bottom, ks, stride=1):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)

# main netspec wrapper
def caffenet_multilabel(data_layer_params, datalayer):
    # setup the python data layer 
    n = caffe.NetSpec()
    #n.data, n.label = L.Python(module = 'pascal_multilabel_datalayers', layer = datalayer, 
    #                           ntop = 2, param_str=str(data_layer_params))
    n.data, n.label = L.Python(module = 'itugo_doudou_multilabel_datalayers', layer = datalayer, 
                               ntop = 2, param_str=str(data_layer_params))

    # the net itself
    n.conv1, n.relu1 = conv_relu(n.data, 11, 96, stride=4)
    n.pool1 = max_pool(n.relu1, 3, stride=2)
    n.norm1 = L.LRN(n.pool1, local_size=5, alpha=1e-4, beta=0.75)
    n.conv2, n.relu2 = conv_relu(n.norm1, 5, 256, pad=2, group=2)
    n.pool2 = max_pool(n.relu2, 3, stride=2)
    n.norm2 = L.LRN(n.pool2, local_size=5, alpha=1e-4, beta=0.75)
    n.conv3, n.relu3 = conv_relu(n.norm2, 3, 384, pad=1)
    n.conv4, n.relu4 = conv_relu(n.relu3, 3, 384, pad=1, group=2)
    n.conv5, n.relu5 = conv_relu(n.relu4, 3, 256, pad=1, group=2)
    n.pool5 = max_pool(n.relu5, 3, stride=2)
    n.fc6, n.relu6 = fc_relu(n.pool5, 4096)
    n.drop6 = L.Dropout(n.relu6, in_place=True)
    n.fc7, n.relu7 = fc_relu(n.drop6, 4096)
    n.drop7 = L.Dropout(n.relu7, in_place=True)
    n.score = L.InnerProduct(n.drop7, num_output=LABEL_NUM)
    n.loss = L.SigmoidCrossEntropyLoss(n.score, n.label)
    
    return str(n.to_proto())

def hamming_distance(gt, est):
    return sum([1 for (g, e) in zip(gt, est) if g == e]) / float(len(gt))

def check_accuracy(net, num_batches, batch_size = 128):
    acc = 0.0
    for t in range(num_batches):
        net.forward()
        gts = net.blobs['label'].data
        ests = net.blobs['score'].data > 0
        for gt, est in zip(gts, ests): #for each ground truth and estimated label vector
            acc += hamming_distance(gt, est)
    return acc / (num_batches * batch_size)

def check_baseline_accuracy(net, num_batches, batch_size = 128):
    acc = 0.0
    for t in range(num_batches):
        net.forward()
        gts = net.blobs['label'].data
        ests = np.zeros((batch_size, len(gts)))
        for gt, est in zip(gts, ests): #for each ground truth and estimated label vector
            acc += hamming_distance(gt, est)
    return acc / (num_batches * batch_size)


if __name__ == '__main__': 

    # make sure we have the caffenet weight downloaded.
    if not os.path.isfile(caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'):
        print("Please Download pre-trained CaffeNet model...")

    # initialize caffe for gpu mode
    caffe.set_mode_gpu()
    caffe.set_device(0)

    #crete net and solver prototxts
    #workdir = './pascal_multilabel_with_datalayer' #voc2007
    workdir = './itugo_doudou_multilabel_with_datalayer' #itugo_doudou
    if not os.path.isdir(workdir):
        os.makedirs(workdir)

    solverprototxt = tools.CaffeSolver(trainnet_prototxt_path = osp.join(workdir, "trainnet.prototxt"), testnet_prototxt_path = osp.join(workdir, "valnet.prototxt"))
    solverprototxt.sp['display'] = "10"
    solverprototxt.sp['base_lr'] = "0.01"
    solverprototxt.write(osp.join(workdir, 'solver.prototxt'))

    # write train net.
    with open(osp.join(workdir, 'trainnet.prototxt'), 'w') as f:
        # provide parameters to the data layer as a python dictionary. Easy as pie!
        data_layer_params = dict(batch_size = 128, im_shape = [227, 227], split = 'train', pascal_root = pascal_root)
        #f.write(caffenet_multilabel(data_layer_params, 'PascalMultilabelDataLayerSync'))
        f.write(caffenet_multilabel(data_layer_params, 'Itugo_Doudou_MultilabelDataLayerSync'))

    # write validation net.
    with open(osp.join(workdir, 'valnet.prototxt'), 'w') as f:
        data_layer_params = dict(batch_size = 128, im_shape = [227, 227], split = 'val', pascal_root = pascal_root)
        #f.write(caffenet_multilabel(data_layer_params, 'PascalMultilabelDataLayerSync'))
        f.write(caffenet_multilabel(data_layer_params, 'Itugo_Doudou_MultilabelDataLayerSync'))

    #fineturn model
    solver = caffe.SGDSolver(osp.join(workdir, 'solver.prototxt'))
    solver.net.copy_from(caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel')
    solver.test_nets[0].share_with(solver.net)
    solver.step(1)
    #solver.snapshot()

    #show GT
    #transformer = tools.SimpleTransformer() # This is simply to add back the bias, re-shuffle the color channels to RGB, and so on...
    #image_index = 0 # First image in the batch.
    #plt.figure()
    #plt.imshow(transformer.deprocess(copy(solver.net.blobs['data'].data[image_index, ...])))
    #gtlist = solver.net.blobs['label'].data[image_index, ...].astype(np.int)
    #plt.title('GT: {}'.format(classes[np.where(gtlist)]))
    #plt.axis('off');

    #train
    for itt in range(10):
        solver.step(1000)
        solver.snapshot()
        print 'itt:{:3d}'.format((itt + 1) * 1000), 'accuracy:{0:.4f}'.format(check_accuracy(solver.test_nets[0], 50))
    print 'Baseline accuracy:{0:.4f}'.format(check_baseline_accuracy(solver.test_nets[0], 1000/128))
    #solver.snapshot()

    #test
    #test_net = solver.test_nets[0]
    #for image_index in range(5):
    #    plt.figure()
    #    plt.imshow(transformer.deprocess(copy(test_net.blobs['data'].data[image_index, ...])))
    #    gtlist = test_net.blobs['label'].data[image_index, ...].astype(np.int)
    #    estlist = test_net.blobs['score'].data[image_index, ...] > 0
    #    plt.title('GT: {} \n EST: {}'.format(classes[np.where(gtlist)], classes[np.where(estlist)]))
    #    plt.axis('off')

    print 'Done!!'





