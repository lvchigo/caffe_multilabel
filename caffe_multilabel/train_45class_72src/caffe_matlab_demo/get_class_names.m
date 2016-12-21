function [names, cnames] = get_class_names_style(names_list, cnames_list)


%cnames_list = '/home/tangyuan/Data/model/finetune_flickr_style/cnames';
%names_list = '/home/tangyuan/Data/model/finetune_flickr_style/names';



fptr_cls = fopen(names_list, 'r');
fptr_clsc = fopen(cnames_list, 'r');
cnames = {};
names = {};
while ~feof(fptr_cls) && ~feof(fptr_clsc)

cnames = [cnames; char(fgetl(fptr_clsc))];
names = [names; char(fgetl(fptr_cls))];
   
end
fclose(fptr_cls);
fclose(fptr_clsc);
