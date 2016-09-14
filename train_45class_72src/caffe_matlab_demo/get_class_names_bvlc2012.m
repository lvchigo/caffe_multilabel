function cnames = get_class_names_bvlc2012()


names_list = '/home/public/model/bvlc_reference_caffenet/synset_words.txt';


fptr_cls = fopen(names_list, 'r');
cnames = {};
while ~feof(fptr_cls)
	   %vec = strsplit(fgetl(fptr_cls), ' ');
vec = fgetl(fptr_cls);
vec = vec(11:end);
cnames = [cnames; vec];
    
end
fclose(fptr_cls)
