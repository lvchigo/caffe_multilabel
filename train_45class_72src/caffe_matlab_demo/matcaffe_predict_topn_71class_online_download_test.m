function matcaffe_predict_topn_71class_online_download_test()

clear all ;clc;

%% caffe init
addpath /home/chigo/caffe_tts/caffe-master/matlab/caffe
addpath(genpath('/home/chigo/toolbox/imageclassification'));

opts.model_def_file = '../deploy.prototxt';
opts.model_file = '../model1/caffenet_train_iter_3000.caffemodel';
opts.mean_file = 'train_mean_71class_256.mat';
opts.image_dim = 256;
opts.cropped_dim = 227;
opts.flip_dim = 10;
opts.use_gpu = 1;

matcaffe_init_opt(opts.use_gpu, opts.model_def_file, opts.model_file);

%% do job
T = 0.9;
N_T = 2;
N_Label = 71;
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
    'people.self' 'people.street'...
    'pet.alpaca' 'pet.cat' 'pet.dog' 'pet.rabbit'...
    'scene.clothingshop' 'scene.courtyard' 'scene.desert' 'scene.forest' 'scene.grasslands'...
    'scene.handdrawn.color' 'scene.handdrawn.whiteblack' 'scene.highway' 'scene.house' 'scene.mountain' 'scene.sea'...
    'scene.sky' 'scene.sticker' 'scene.street' 'scene.supermarket' 'scene.tallbuilding'...
    'scene.text'};

image_save_dir = '../online_download_img_test_50w/';
image_save_dir_src = '../online_download_img/';

%% clear file && mkdir
rm_img_save_dir=sprintf('rm -rf %s/',image_save_dir );
mk_img_save_dir=sprintf('mkdir %s/',image_save_dir );
dos(rm_img_save_dir);
dos(mk_img_save_dir);

rm_img_save_dir=sprintf('rm -rf %s/',image_save_dir_src );
mk_img_save_dir=sprintf('mkdir %s/',image_save_dir_src );
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

%% read img
start_id = 134800001;%20150326
total = 500000;
num_load = 0;
for i = start_id:start_id+total
	  id = num2str(i);
	  imurl = get_in_imurl(id);
      if length(imurl) < 16
          continue;
      end
      
      %imf = t.url, %[t.url, '?imageMogr2/format/jpg/thumbnail/256x256/quality/80!'];
      imf = fullfile(image_save_dir, [id, '.jpg']);
      try
          dos(['wget -T 3 -t 3 -q ', imurl, ' -O ', imf]);
      catch
          continue;
      end   
      
      try
          im = imread(imf);
      catch
          delete(imf);
          continue;
      end      
      
      if size(im, 3)<3
          im = cat(3, im,im,im);
      end
      im = imresize(im, [256,256]);

      num_load = num_load + 1;
      if (rem(num_load,5) == 0)
          fprintf('Load %d img...\n',num_load);
      end
    
    scores = matcaffe_predict(im, opts);
    [val, idx] = sort(scores, 'descend');
          
    predicted_label = 0;
    for m = 1:N_T
        T_CH = T - 0.1*(m-1);
        
	    % save file
        if ( val(1)>T_CH )
            predicted_label = idx(1);
            savefile=sprintf('%s/%.1f_%d_%s/%d_%s_%.2f_20150401_%ld.jpg',image_save_dir,...
                T_CH,predicted_label,char(Label{predicted_label}),...
                predicted_label,char(Label{predicted_label}),val(1),num_load);           
            imwrite(im,savefile);
        end      
    end
    
    %savefile=sprintf('%s/%ld.jpg',image_save_dir_src,num_load);           
    %imwrite(im,savefile);
    
    try
        delete(imf);
    catch
        continue;
    end
end

fprintf('All load %d img!\n',i);
clear all ;


