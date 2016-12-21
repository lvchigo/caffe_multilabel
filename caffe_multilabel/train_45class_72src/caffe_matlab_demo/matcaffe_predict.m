function [scores, maxlabel] = matcaffe_predict(im, opts)
  % scores = matcaffe_demo(im, use_gpu)
%
% Demo of the matlab wrapper using the ILSVRC network.
%
% input
%   im       color image as uint8 HxWx3
  %   use_gpu  1 to use the GPU, 0 to use the CPU
%
% output
%   scores   1000-dimensional ILSVRC score vector
%
  % You may need to do the following before you start matlab:
  %  $ export LD_LIBRARY_PATH=/opt/intel/mkl/lib/intel64:/usr/local/cuda-5.5/lib64
%  $ export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
% Or the equivalent based on where things are installed on your system
%
			 % Usage:
  %  im = imread('../../examples/images/cat.jpg');
%  scores = matcaffe_demo(im, 1);
%  [score, class] = max(scores);
% Five things to be aware of:
%   caffe uses row-major order
%   matlab uses column-major order
%   caffe uses BGR color channel order
%   matlab uses RGB color channel order
%   images need to have the data mean subtracted

% Data coming in from matlab needs to be in the order 
%   [width, height, channels, images]
% where width is the fastest dimension.
% Here is the rough matlab for putting image data into the correct
% format:
%   % convert from uint8 to single
%   im = single(im);
%   % reshape to a fixed size (e.g., 227x227)
%   im = imresize(im, [IMAGE_DIM IMAGE_DIM], 'bilinear');
%   % permute from RGB to BGR and subtract the data mean (already in BGR)
%   im = im(:,:,[3 2 1]) - data_mean;
%   % flip width and height to make width the fastest dimension
%   im = permute(im, [2 1 3]);

% If you have multiple images, cat them with cat(4, ...)

% The actual forward function. It takes in a cell array of 4-D arrays as
% input and outputs a cell array. 




% prepare oversampled input
% input_data is Height x Width x Channel x Num
  input_data = {prepare_image(im, opts)};


% do forward pass to get scores
% scores are now Width x Height x Channels x Num
%       tic;
scores = caffe('forward', input_data);
%toc;

scores = scores{1};
%size(scores)
scores = squeeze(scores);
scores = mean(scores,2);

[~,maxlabel] = max(scores);

% ------------------------------------------------------------------------
function images = prepare_image(im, opts)
% ------------------------------------------------------------------------
% ilsvrc_2012_mean
% /home/tangyuan/Data/model/placesCNN/places_mean.mat
  d = load(opts.mean_file);
IMAGE_MEAN = d.image_mean;
IMAGE_DIM = opts.image_dim;
CROPPED_DIM = opts.cropped_dim;
FLIP_DIM = opts.flip_dim;

% resize to fixed input size
im = single(im);
im = imresize(im, [IMAGE_DIM IMAGE_DIM], 'bilinear');
% permute from RGB to BGR (IMAGE_MEAN is already BGR)
im = im(:,:,[3 2 1]) - IMAGE_MEAN;

% oversample (4 corners, center, and their x-axis flips) 10 256
images = zeros(CROPPED_DIM, CROPPED_DIM, 3, FLIP_DIM, 'single');

indices = [0 IMAGE_DIM-CROPPED_DIM] + 1;
curr = 1;
for i = indices
  for j = indices
  images(:, :, :, curr) = ...
  permute(im(i:i+CROPPED_DIM-1, j:j+CROPPED_DIM-1, :), [2 1 3]);
images(:, :, :, curr+FLIP_DIM/2) = images(end:-1:1, :, :, curr);
curr = curr + 1;
  end
end

  center = floor(indices(2) / 2)+1;
images(:,:,:,FLIP_DIM/2) = ...
  permute(im(center:center+CROPPED_DIM-1,center:center+CROPPED_DIM-1,:), ...
	  [2 1 3]);
images(:,:,:,FLIP_DIM) = images(end:-1:1, :, :, curr);

