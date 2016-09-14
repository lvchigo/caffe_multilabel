function matcaffe_predict_topn()

clear all ;clc;
%% caffe init
addpath /home/chigo/caffe_tts/caffe-master/matlab/caffe
addpath(genpath('/home/chigo/toolbox/imageclassification'));

opts.model_def_file = '../deploy.prototxt';
opts.model_file = '../model1/caffenet_train_iter_34000.caffemodel';
opts.mean_file = 'train_mean_62class_256.mat';
opts.image_dim = 256;
opts.cropped_dim = 227;
opts.flip_dim = 10;
opts.use_gpu = 1;

matcaffe_init_opt(opts.use_gpu, opts.model_def_file, opts.model_file);


%% do job
topN = 1;
T = 0.9;

image_dir = '/home/chigo/image/image1/test';
image_list = '../img_test.txt';
image_save_dir = '../save/';

imdb = get_image_db(image_dir, image_list);
cnames = imdb.cnames;
nclasses = length(cnames);
cprob = zeros(1, nclasses);

fprintf('All Load Label:');
for i = 1:nclasses
    %fprintf('"%s" ',char(cnames{i}));
    fprintf('%d-%s,',i,char(cnames{i}));
end
fprintf('\n');

truth_label = imdb.labels;
predicted_label = zeros(size(truth_label));

%% clear file && mkdir
rm_img_save_dir=sprintf('rm -rf %s/',image_save_dir);
rm_img_save_dir_up=sprintf('rm -rf %s/up/',image_save_dir);
mk_img_save_dir=sprintf('mkdir %s/',image_save_dir);
mk_img_save_dir_up=sprintf('mkdir %s/up/',image_save_dir);
dos(rm_img_save_dir);
dos(rm_img_save_dir_up);
dos(mk_img_save_dir);
dos(mk_img_save_dir_up);

%% count Result
num_load_img = zeros(nclasses);
num_T_img = zeros(nclasses);
num_err_img = zeros(nclasses);

%% read img
for i = 1:length(imdb.pathes)
    
    if (rem(i,5) == 0)
        fprintf('Load %d img...\n',i);
    end
    
    imf = imdb.pathes{i};
    im = imread(imf);   
    
    if size(im, 3) < 3
        im = cat(3, im, im, im);
    end
    
    scores = matcaffe_predict(im, opts);
    [val, idx] = sort(scores, 'descend');
    
    c = truth_label(i);
    prob = ismember(idx(1:topN), c);
    if topN == 1
        predicted_label(i) = idx(1);
    else
        
        cdx = find(prob == 1);
        if length(cdx) == 1
            predicted_label(i) = idx(cdx(1));
        else
            predicted_label(i) = idx(1);
        end
    end
    
    name_truth_label = char(cnames{truth_label(i)});
    name_predicted_label = char(cnames{predicted_label(i)});

    % save file
    if ( val(1)>T )
        if ( truth_label(i)==predicted_label(i) )
            savefile=sprintf('%s/up/%d_%s_%.2f_%d.jpg',image_save_dir,...
                predicted_label(i),name_predicted_label,val(1),i);
            num_T_img(predicted_label(i)) = num_T_img(predicted_label(i)) + 1;
        else
            savefile=sprintf('%s/up/err_%d_%s_%d_%s_%.2f_%d.jpg',image_save_dir,...
                truth_label(i),name_truth_label,predicted_label(i),name_predicted_label,val(1),i);
            num_err_img(predicted_label(i)) = num_err_img(predicted_label(i)) + 1;
        end
        imwrite(im,savefile);
    end
    num_load_img(truth_label(i)) = num_load_img(truth_label(i)) + 1;   
end

%% get Result
Recall = zeros(nclasses);
Precision = zeros(nclasses);
num_all_load_img = 0;
mean_Recall = 0.0;
mean_Precision = 0.0;
num_load_class = 0;
for i = 1:nclasses
    if ( (num_load_img(i)>0) && ((num_T_img(i) + num_err_img(i))>0) )
        Recall(i) = num_T_img(i)*1.0/num_load_img(i);
        Precision(i) = num_T_img(i)*1.0/(num_T_img(i) + num_err_img(i));
        if (Precision(i)>0)
            fprintf('class:%d-%s,num:%d,Recall:%.4f,Precision:%.4f\n',...
                i,char(cnames{i}),num_load_img(i),Recall(i),Precision(i));
        end
        num_all_load_img = num_all_load_img + num_load_img(i);
        mean_Recall = mean_Recall + Recall(i);
        mean_Precision = mean_Precision + Precision(i);
        num_load_class = num_load_class + 1;
    end
end
mean_Recall = mean_Recall*1.0/num_load_class;
mean_Precision = mean_Precision*1.0/num_load_class;
fprintf('all load %d img,all load %d class,Recall:%.4f,Precision:%.4f\n',...
    num_all_load_img,num_load_class,mean_Recall,mean_Precision);

%% calc_confusion_matrix
confusion_matrix = calc_confusion_matrix(truth_label, predicted_label);
mean(diag(confusion_matrix));
accs = diag(confusion_matrix);

