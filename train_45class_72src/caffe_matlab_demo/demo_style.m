function demo_style

addpath /home/public/projects/caffe/matlab/caffe

cnames_list = '/home/public/model/finetune_flickr_style/cnames';
names_list = '/home/public/model/finetune_flickr_style/names';
[names, cnames] = get_class_names(names_list, cnames_list);

im = imread('http://farm8.staticflickr.com/7346/10101206875_c9d243c583.jpg');
%imshow(im)
length(cnames)

opts.model_def_file = '/home/public/model/finetune_flickr_style/deploy.prototxt';
opts.model_file = '/home/public/model/finetune_flickr_style/finetune_flickr_style.caffemodel';
opts.mean_file = '/home/public/model/bvlc_reference_caffenet/ilsvrc_2012_mean.mat';
opts.image_dim = 256;
opts.cropped_dim = 227;
opts.flip_dim = 10;
opts.use_gpu = 1;

matcaffe_init_opt(opts.use_gpu, opts.model_def_file, opts.model_file);


rt = tic;
[scores, maxlabel] = matcaffe_predict(im, opts);
toc(rt)
  
[val, idx] = sort(scores, 'descend');
val(1:3)
cnames{idx(1:3)}