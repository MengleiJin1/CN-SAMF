
% run_tracker.m

close all;
% clear all;

%choose the path to the videos (you'll be able to choose one with the GUI)
base_path = 'F:/徐福来/野外观测/database/benchmark50';
base_path ='F:\徐福来\野外观测\database\新建文件夹\benchmark50';
% base_path = 'F:/毕业资料/目标跟踪/测试视频序列';

%parameters according to the paper
params.padding =1.5;         			   % extra area surrounding the target
params.output_sigma_factor = 1/16;		   % spatial bandwidth (proportional to target)
params.sigma = 0.2;         			   % gaussian kernel bandwidth
params.lambda = 1e-2;					   % regularization (denoted "lambda" in the paper)
params.learning_rate = 0.012;			   % learning rate for appearance model update scheme (denoted "gamma" in the paper)
params.compression_learning_rate = 0.015;   % learning rate for the adaptive dimensionality reduction (denoted "mu" in the paper)
%params.non_compressed_features = {'gray'}; % features that are not compressed, a cell with strings (possible choices: 'gray', 'cn')
%params.compressed_features = {'cn'};       % features that are compressed, a cell with strings (possible choices: 'gray', 'cn')
params.num_compressed_dim =100;             % the dimensionality of the compressed features
params.hog_orientations = 9;   %新
params.cell_size = 4;           %新
params.visualization = 1;

%ask the user for the video
[video_path,video_name] = choose_video(base_path);
if isempty(video_path), return, end  %user cancelled
[img_files, pos, target_sz, ground_truth, video_path] = ...
	load_video_info(video_path);

params.init_pos = floor(pos) + floor(target_sz/2);
params.wsize = floor(target_sz);
params.img_files = img_files;
params.video_path = video_path;

[positions, fps] = color_tracker(params);

outname2 = './result/';
result = [positions(:,[2,1]) - (positions(:,[4,3]) - 1) / 2 , positions(:,[4,3])];
dlmwrite([outname2 video_name  '_Ours.txt'],result,'delimiter',',','newline','pc');
% calculate precisions
[distance_precision, PASCAL_precision, average_center_location_error] = ...
    compute_performance_measures(positions, ground_truth);

fprintf('Center Location Error: %.3g pixels\nDistance Precision: %.3g %%\nOverlap Precision: %.3g %%\nSpeed: %.3g fps\n', ...
    average_center_location_error, 100*distance_precision, 100*PASCAL_precision, fps);
