function matcaffe_predict_topn_72class_onlinetest()

clear all ;clc;

%% caffe init
addpath /home/chigo/caffe_tts/caffe-master/matlab/caffe
addpath(genpath('/home/chigo/toolbox/imageclassification'));

opts.model_def_file = '../deploy.prototxt';
opts.model_file = '../model_72/caffenet_train_iter_1000.caffemodel';
%opts.model_file = '../model_fineturn_1/caffenet_train_iter_8000.caffemodel';
opts.mean_file = 'train_mean_72class_256.mat';
opts.image_dim = 256;
opts.cropped_dim = 227;
opts.flip_dim = 10;
opts.use_gpu = 1;

matcaffe_init_opt(opts.use_gpu, opts.model_def_file, opts.model_file);

%% do job
T = 0.9;
N_T = 2;
N_Label = 72;
Label = {'food.barbecue' 'food.bread' 'food.cake' 'food.candy' 'food.coffee'...
    'food.cook' 'food.cookie' 'food.crab' 'food.dumpling' 'food.fruit'...
    'food.hamburger' 'food.hotpot' 'food.icecream' 'food.pasta' 'food.pizza'...
    'food.rice' 'food.steak' 'food.sushi'...
    'goods.airplane' 'goods.bag' 'goods.bangle' 'goods.bottle' 'goods.bracelet'...
    'goods.camera' 'goods.car' 'goods.clothes' 'goods.cosmetics' 'goods.drawbar'...
    'goods.flower' 'goods.glass' 'goods.guitar' 'goods.hat' 'goods.laptop'...
    'goods.lipstick' 'goods.manicure' 'goods.pendant' 'goods.phone' 'goods.puppet'...
    'goods.ring' 'goods.ship' 'goods.shoe' 'goods.train' 'goods.watch'... 
    'people.eye' 'people.friend' 'people.hair' 'people.kid' 'people.lip'...
    'people.self.female' 'people.self.male' 'people.street'...
    'pet.alpaca' 'pet.cat' 'pet.dog' 'pet.rabbit'...
    'scene.clothingshop' 'scene.courtyard' 'scene.desert' 'scene.forest' 'scene.grasslands'...
    'scene.handdrawn.color' 'scene.handdrawn.whiteblack' 'scene.highway' 'scene.house' 'scene.mountain' 'scene.sea'...
    'scene.sky' 'scene.sticker' 'scene.street' 'scene.supermarket' 'scene.tallbuilding'...
    'scene.text'};

image_list = '/home/chigo/image/test/list_test0313_10k.txt';%test
%image_list = '../img_online_133060001_4w.txt';
%image_list = '/home/chigo/image/img_download/list_add_2015031701.txt';%other-3w
image_save_dir = '../save/';

fid = fopen(image_list);
fileLines = textscan(fid, '%s');
fclose(fid);

files = fileLines{1, 1};
file_count = numel(files);
fprintf('Found %d files\n', file_count);

%% clear file && mkdir
rm_img_save_dir=sprintf('rm -rf %s',image_save_dir);
mk_img_save_dir=sprintf('mkdir %s',image_save_dir);
dos(rm_img_save_dir);
dos(mk_img_save_dir);
       
for m = 1:N_T
    T_CH = T - 0.1*(m-1);
    for i = 1:N_Label
        rm_img_save_dir=sprintf('rm -rf %s/%.1f_%d_%s/',image_save_dir,T_CH,i,char(Label{i}) );
        mk_img_save_dir=sprintf('mkdir %s/%.1f_%d_%s/',image_save_dir,T_CH,i,char(Label{i}));
        dos(rm_img_save_dir);
        dos(mk_img_save_dir);
    end
end

%% muti thread
% matlabpool local 2;

%% read img
for i = 1:file_count
% parfor  i = 1:file_count %muti thread
    if (rem(i,5) == 0)
        fprintf('Load %d img...\n',i);
    end
    
    predicted_label = 0;
    im = imread(files{i});   
    if size(im, 3) < 3
        im = cat(3, im, im, im);
    end
    im = imresize(im, [256,256]);
    
    scores = matcaffe_predict(im, opts);
    [val, idx] = sort(scores, 'descend');
          
    for m = 1:N_T
        T_CH = T - 0.1*(m-1);

%         if ( val(1)>T_CH )
%             if ( (idx(1)>=1) && (idx(1)<17) )
%             	predicted_label(i) = 1; %food-1
%             elseif ( ( (idx(1)>=17) && (idx(1)<41) ) || (idx(1)==66) || (idx(1)==71) || (idx(1)==75) )
%             	predicted_label(i) = 2; %goods-2-other
%             elseif ( (idx(1)>=41) && (idx(1)<48) )
%             	predicted_label(i) = 3; %people-3
%             elseif ( (idx(1)>=48) && (idx(1)<63) )
%             	predicted_label(i) = 4; %pet-4
%             elseif ( ( (idx(1)>=63) && (idx(1)<66) ) || ...
%                 ( (idx(1)>=67) && (idx(1)<71) ) ||...
%                 ( (idx(1)>=72) && (idx(1)<75) ) )
%             	predicted_label(i) = 5; %scene-5
%             end
%         else
%             predicted_label = 2;
%         end

        % save file
        if ( val(1)>T_CH )
            predicted_label = idx(1);
            savefile=sprintf('%s/%.1f_%d_%s/%d_%s_%.2f_%d.jpg',image_save_dir,...
                T_CH,predicted_label,char(Label{predicted_label}),...
                predicted_label,char(Label{predicted_label}),val(1),i);  
            imwrite(im,savefile); 
        end
        
    end
end

%% muti thread
% matlabpool close

fprintf('All load %d img!\n',i);
clear all ;


