function imurl = get_in_imurl(id)

%imf = fullfile(out_dir, [id, '.jpg'])
%dos(['wget -T 3 -t 3 ', imurl, ' -O ', imf]);
% do ...
% delete(imf);

base_url = 'http://in.itugo.com/api/getphotourl?id=';
%id = '86717488';
imurl = '';
T = 5;

try
query_url = [base_url, id];
  html = urlread(query_url, 'Timeout', T); %'Charset','UTF-8
data = parse_json(html);
imurl = data{1,1}.data;
catch
end

if length(imurl) < 16
    imurl = '';
end
