%% This script calculates the mean image!

clear all ;clc;
%% Configuration

config = struct();

config.base_address = '/home/chigo/image/image1/val/';
config.target_file  = '../img_val.txt';
config.target_size  = [256 256];

config.result_file  = 'train_mean_72class_256.mat';

%% Doing the job!

fid = fopen(config.target_file);
%% 

fileLines = textscan(fid, '%[^ ] %d', 'delimiter', '\n', 'BufSize', 10000);
fclose(fid);

files = fileLines{1, 1};
file_count = numel(files);
fprintf('Found %d files\n', file_count);

images = cell(file_count, 1);
fprintf('Processing the files ...\n'); tic;
parfor f = 1:file_count,
    fprintf('Doing %s\n', files{f});
    im = imread([config.base_address files{f}]);
    %im = imread(files{f}); %full-path
    im = imresize(im, config.target_size, 'bilinear');
    images{f} = im;
end
fprintf(' done in %.2fs\n', toc);

%% The rest of the job
fprintf('Calculating the mean\n');

images_reshape = cellfun(@(x) reshape(x, [1 256 256 3]), images, 'UniformOutput', false);
%images_reshape = cellfun(@(x) reshape(x, [1 config.target_size 3]), images, 'UniformOutput', false);
images_reshape = cell2mat(images_reshape);
image_mean_exp = mean(images_reshape);

image_mean = single(squeeze(image_mean_exp));

%% Saving the output
fprintf('Saving the file ...\n'); tic;
save(config.result_file, 'image_mean');
%% SECTION TITLE
% DESCRIPTIVE TEXT
fprintf(' done in %.2fs\n', toc);

%% clear up
clear image_mean images_reshape image_mean_exp images;
clear all ;clc;
