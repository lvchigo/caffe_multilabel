function imdb = get_image_db(image_dir, image_list)

%image_dir = '/home/xiaogao/matlab_test_20150306/test';
%image_list = '/home/xiaogao/matlab_test_20150306/img_test.txt';

fptr = fopen(image_list, 'r');

cnames = {};
labels = [];
path = {};
while ~feof(fptr)

linestr = fgetl(fptr);
cellstr = regexp(linestr, ' ', 'split');
imstr = cellstr{1};
idstr = cellstr{2};
cellstr = regexp(imstr, '/', 'split');
cstr = cellstr{1};

labels(end+1) = str2num(idstr)+1;
path{end+1} = fullfile(image_dir, char(imstr));
cnames{str2num(idstr)+1} = cstr;

end

imdb.cnames = cnames;
imdb.labels = labels;
imdb.pathes = path;
