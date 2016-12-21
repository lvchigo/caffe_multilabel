function  matcaffe_init_opt(use_gpu, model_def_file, model_file)
% matcaffe_init(model_def_file, model_file, use_gpu)
% Initilize matcaffe wrapper

if caffe('is_initialized') == 0
  if exist(model_file, 'file') == 0
  % NOTE: you'll have to get the pre-trained ILSVRC network
    error('You need a network model file');
  end
  if ~exist(model_def_file,'file')
    % NOTE: you'll have to get network definition
	    error('You need the network prototxt definition');
  end
  caffe('init', model_def_file, model_file, 'test')
end
  fprintf('Done with init\n');

% set to use GPU or CPU
if use_gpu
fprintf('Using GPU Mode\n');
caffe('set_mode_gpu');
 else
   fprintf('Using CPU Mode\n');
caffe('set_mode_cpu');
end
fprintf('Done with set_mode\n');

% put into test mode
%caffe('set_phase_test');
%fprintf('Done with set_phase_test\n');
