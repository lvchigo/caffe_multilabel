function demo_in75c

addpath /home/public/projects/caffe/matlab/caffe


im = imread('/home/xiaogao/matlab_test_20150306/test/food.barbecue/food.barbecue437.jpg');
%imshow(im)

opts.model_def_file = '/home/xiaogao/matlab_test_20150306/deploy.prototxt';
opts.model_file = '/home/xiaogao/matlab_test_20150306/caffenet_train_iter_22000.caffemodel';
opts.mean_file = '/home/public/model/in75c_mean.mat';
opts.image_dim = 256;
opts.cropped_dim = 227;
opts.flip_dim = 10;
opts.use_gpu = 1;

matcaffe_init_opt(opts.use_gpu, opts.model_def_file, opts.model_file);


rt = tic;
[scores, maxlabel] = matcaffe_predict(im, opts);
toc(rt)
  
[val, idx] = sort(scores, 'descend');
[idx(1:3) val(1:3) ]

