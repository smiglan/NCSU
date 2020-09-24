
RTF_file_path = "C:\Users\jjreese\Desktop\Error_Light.rtf";
fid = fopen(RTF_file_path);
time = [];
offset = [];
%get each line in the file and run WHILE loop until you read the line with
%'cells_QC'
while ~feof(fid)
    tline = fgetl(fid);
    x = strsplit(tline,",");
    time = [time str2double(x(1))];
    offset = [offset str2double(x(2))];
end


hold on;
zero = zeros(1,sz);
plot(time,offset)
[blah,sz] = size(time);

plot(time,zeros(1,sz))
title('Offset')

hold off;


% error_array = offset - zero;
% subplot(2,1,2)
% plot(time,error_array);
% 
% title('Error')

fclose(fid);