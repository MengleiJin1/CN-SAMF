function [img_files, pos, target_sz, ground_truth, video_path] = load_video_info(video_path)

% [img_files, pos, target_sz, ground_truth, video_path] = load_video_info(video_path)

% text_files = dir([video_path '*_frames.txt']);
% f = fopen([video_path text_files(1).name]);
% frames = textscan(f, '%f,%f');
% fclose(f);

text_files = dir([video_path 'groundtruth_rect.txt']);
assert(~isempty(text_files), 'No initial position and ground truth (*_gt.txt) to load.')

f = fopen([video_path text_files(1).name]);
try
		ground_truth = textscan(f, '%f,%f,%f,%f', 'ReturnOnError',false);  
	catch  %#ok, try different format (no commas)
		frewind(f);%回到开始指针
		ground_truth = textscan(f, '%f %f %f %f');  
	end
%ground_truth = textscan(f, '%f,%f,%f,%f');  %[x, y, width, height]
ground_truth = cat(2, ground_truth{:});
fclose(f);

%set initial position and size
target_sz = [ground_truth(1,4), ground_truth(1,3)];
pos = [ground_truth(1,2), ground_truth(1,1)];

ground_truth = [ground_truth(:,[2,1]) + (ground_truth(:,[4,3]) - 1) / 2 , ground_truth(:,[4,3])];
  video_path = [video_path 'img/'];
  img_files = dir([video_path '*.png']);
		if isempty(img_files),
			img_files = dir([video_path '*.jpg']);
			assert(~isempty(img_files), 'No image files to load.')
		end
img_files = sort({img_files.name});
% %see if they are in the 'imgs' subfolder or not
% if exist([video_path num2str(frames{1}, '%04i.png')], 'file'),
%     img_files = num2str((frames{1} : frames{2})', '%04i.png');
% elseif exist([video_path num2str(frames{1}, '%04i.jpg')], 'file'),
%     img_files = num2str((frames{1} : frames{2})', '%04i.jpg');
% elseif exist([video_path num2str(frames{1}, '%04i.bmp')], 'file'),
%     img_files = num2str((frames{1} : frames{2})', '%04i.bmp');
% else
%     error('No image files to load.')
% % end
% 
% %list the files
% img_files = cellstr(img_files);

end

