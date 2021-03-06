function matcaffe_predict_topn_16classfrom41class_online_download_test()

clear all ;clc;

%% caffe init
addpath /home/chigo/caffe_tts/caffe-master/matlab/caffe
addpath(genpath('/home/chigo/toolbox/imageclassification'));

opts.model_def_file = '../deploy.prototxt';
opts.model_file = '../model3/caffenet_train_iter_47000.caffemodel';
opts.mean_file = 'train_mean_41class_256.mat';
opts.image_dim = 256;
opts.cropped_dim = 227;
opts.flip_dim = 10;
opts.use_gpu = 1;

matcaffe_init_opt(opts.use_gpu, opts.model_def_file, opts.model_file);

%% do job
T = 0.9;
N_T = 2;
% N_Label = 41;
% Label = {'food.coffee' 'food.dumpling' 'food.hamburger' 'food.rice' 'food.sushi'...
%     'goods.airplane' 'goods.bag' 'goods.bangle' 'goods.bottle' 'goods.bracelet'...
%     'goods.camera' 'goods.clothes' 'goods.cosmetics' 'goods.drawbar' 'goods.glass'...
%     'goods.guitar' 'goods.hat' 'goods.laptop' 'goods.lipstick' 'goods.manicure'...
%     'goods.pendant' 'goods.phone' 'goods.puppet' 'goods.ring' 'goods.shoe'...
%     'goods.watch' 'people.friend' 'people.hair' 'people.lip' 'people.self'...
%     'pet.cat' 'pet.dog' 'pet.rabbit' 'scene.clothingshop' 'scene.forest'...
%     'scene.handdrawn' 'scene.highway' 'scene.sticker' 'scene.street' 'scene.tallbuilding'...
%     'scene.text'};
N_Label = 16;
Label = {'food' 'people' 'pet' 'scene'...
    'goods.bag' 'goods.clothes' 'goods.cosmetics' 'goods.glass'...
    'goods.hair' 'goods.handdrawn' 'goods.hat' 'goods.jewelry'...
    'goods.manicure' 'goods.puppet' 'goods.shoe' 'goods.watch'...
    };

image_save_dir = '/home/chigo/image/online_download_img_test/';

%% clear file && mkdir
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
start_id = 125000000;
total = 200000;
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
      %size(im)

      num_load = num_load + 1;
      if (rem(num_load,5) == 0)
          fprintf('Load %d img...\n',num_load);
      end
    
    scores = matcaffe_predict(im, opts);
    [val, idx] = sort(scores, 'descend');
          
    predicted_label = 0;
    for m = 1:N_T
        T_CH = T - 0.1*(m-1);

        if ( val(1)>T_CH )
            if ( (idx(1)>=1) && (idx(1)<6) )
            	predicted_label = 1;
            elseif ( (idx(1)==27) || (idx(1)==30) )
            	predicted_label = 2;
            elseif ( (idx(1)>=31) && (idx(1)<34) )
            	predicted_label = 3;
            elseif ( (idx(1)==34) || (idx(1)==35) || (idx(1)==37) || (idx(1)==39) || (idx(1)==40) )
            	predicted_label = 4;
            elseif ( idx(1)==7 )
            	predicted_label = 5;
            elseif ( idx(1)==12 )
            	predicted_label = 6;
            elseif ( (idx(1)==13) || (idx(1)==19) )
            	predicted_label = 7;
            elseif ( idx(1)==15 )
            	predicted_label = 8;
            elseif ( idx(1)==28 )
            	predicted_label = 9;
            elseif ( ( idx(1)==36 ) || (idx(1)==38) )
            	predicted_label = 10;
            elseif ( idx(1)==17 )
            	predicted_label = 11;
            elseif ( (idx(1)==8) || (idx(1)==10) || (idx(1)==21) || (idx(1)==24) )
            	predicted_label = 12;
            elseif ( idx(1)==20 )
            	predicted_label = 13; 
            elseif ( idx(1)==23 )
            	predicted_label = 14;
            elseif ( idx(1)==25 )
            	predicted_label = 15;
            elseif ( idx(1)==26 )
            	predicted_label = 16;
            end           
        end
        
        if ( predicted_label>0 )
            % save file
            savefile=sprintf('%s/%.1f_%d_%s/%d_%s_%.2f_%ld.jpg',image_save_dir,...
                T_CH,predicted_label,char(Label{predicted_label}),...
                predicted_label,char(Label{predicted_label}),val(1),num_load);  
            im = imresize(im, [256,256]);
            imwrite(im,savefile);
        end
        
    end
    
    try
        delete(imf);
    catch
        continue;
    end
end

fprintf('All load %d img!\n',i);

