function [positions, fps] = color_tracker(params)

% [positions, fps] = color_tracker(params)

% parameters
padding = params.padding;
output_sigma_factor = params.output_sigma_factor;
sigma = params.sigma;
lambda = params.lambda;
learning_rate = params.learning_rate;
compression_learning_rate = params.compression_learning_rate;
%non_compressed_features = params.non_compressed_features;
%compressed_features = params.compressed_features;
num_compressed_dim = params.num_compressed_dim;
hog_orientations=params.hog_orientations; %新
cell_size=params.cell_size;

video_path = params.video_path;
img_files = params.img_files;
pos = floor(params.init_pos);
target_sz = floor(params.wsize);

visualization = params.visualization;

num_frames = numel(img_files);

% load the normalized Color Name matrix
addpath('./utility');
temp = load('w2crs');
w2c = temp.w2crs;

%use_dimensionality_reduction = ~isempty(compressed_features);

% window size, taking padding into account
sz = floor(target_sz * (1 + padding));

% desired output (gaussian shaped), bandwidth proportional to target size
%output_sigma = sqrt(prod(target_sz)) * output_sigma_factor;
%[rs, cs] = ndgrid((1:sz(1)) - floor(sz(1)/2), (1:sz(2)) - floor(sz(2)/2));
%y = exp(-0.5 / output_sigma^2 * (rs.^2 + cs.^2));
%yf = single(fft2(y));

% store pre-computed cosine window
%cos_window = single(hann(sz(1)) * hann(sz(2))');
%proportional to target size
output_sigma = sqrt(prod(target_sz)) * output_sigma_factor / cell_size;
yf = fft2(gaussian_shaped_labels(output_sigma, floor(sz / cell_size)));
%store pre-computed cosine window
cos_window = hann(size(yf,1)) * hann(size(yf,2))';	
% to calculate precision
positions = zeros(numel(img_files), 4);

% initialize the projection matrix 论文中的B
%search_size = [1];
%search_size = [1  0.985 0.99 0.995 1.005 1.01 1.015 ];
search_size = [1  0.99 0.995 1.005 1.01  ];
% search_size = [1   0.99 1.01 ];
response = zeros(size(cos_window,1),size(cos_window,2),size(search_size,2));
szid = 1;
projection_matrix = [];

% to calculate fps
time = 0;

for frame = 1:num_frames,
    % load image
    im = imread([video_path img_files{frame}]);
    
    tic;
    
    if frame > 1
        % compute the compressed learnt appearance
        zp = feature_projection(z_fo,z_pca, projection_matrix, cos_window);
        
        % extract the feature map of the local image patch 新改了
         for i=1:size(search_size,2)
            tmp_sz = floor((target_sz * (1 + padding))*search_size(i));
            param0 = [pos(2),pos(1), tmp_sz(2)/sz(2), 0,...
                        tmp_sz(1)/sz(1),0];
            param0 = affparam2mat(param0); 
            patch = uint8(warpimg(double(im), param0, sz));
           [fo, xo_pca ]= get_subwindow(patch, w2c,hog_orientations,cell_size);
        
        % do the dimensionality reduction and windowi
            x = feature_projection(fo,xo_pca, projection_matrix, cos_window);
        
        % calculate the response of the classifier
            kf = fft2(dense_gauss_kernel(sigma, x, zp));
            response(:,:,i) = real(ifft2(alphaf_num .* kf ./ alphaf_den));  %equation for fast detection
         end
        %response = real(ifft2(alphaf_num .* kf ./ alphaf_den));
        [vert_delta,tmp] = find(response == max(response(:)), 1);
        szid = floor((tmp-1)/(size(cos_window,2)))+1;
        horiz_delta = tmp - ((szid -1)* size(cos_window,2));
			if vert_delta > size(x,1) / 2,  %wrap around to negative half-space of vertical axis
				vert_delta = vert_delta - size(x,1);
			end
			if horiz_delta > size(x,2) / 2,  %same for horizontal axis
				horiz_delta = horiz_delta - size(x,2);  
            end
            
            tmp_sz = floor((target_sz * (1 + padding))*search_size(szid));
            current_size = tmp_sz(2)/sz(2);
			pos = pos + current_size*cell_size * [vert_delta - 1, horiz_delta - 1];
        % target location is at the maximum response
%         [row, col] = find(response == max(response(:)), 1);
%         pos = pos - floor(sz/2) + [row, col];
    end
    
    % extract the feature map of the local image patch to train the
    % classifer  新改了
     target_sz = target_sz * search_size(szid);
     tmp_sz = floor((target_sz * (1 + padding)));
     param0 = [pos(2),pos(1), tmp_sz(2)/sz(2), 0,...
                        tmp_sz(1)/sz(2)/(sz(1)/sz(2)),0];
     param0 = affparam2mat(param0); 
     patch = uint8(warpimg(double(im), param0, sz));
     [fo,xo_pca]= get_subwindow(patch, w2c,hog_orientations,cell_size);
    
    if frame == 1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
        % initialize the appearance
        z_fo=fo;
        z_pca = xo_pca;       
        % set number of compressed dimensions to maximum if too many
        num_compressed_dim = min(num_compressed_dim, size(fo, 2));
    else
        % update the appearance
        z_fo = (1 - learning_rate) * z_fo + learning_rate * fo;
        z_pca = (1 - learning_rate) * z_pca + learning_rate * xo_pca;
    end
    
    % if dimensionality reduction is used: update the projection matrix
   % if use_dimensionality_reduction
        % compute the mean appearance
        data_mean = mean(z_fo, 1);
        
        % substract the mean from the appearance to get the data matrix
        data_matrix = bsxfun(@minus, z_fo, data_mean);
        % calculate the covariance matrix
        cov_matrix = 1/(size(z_fo,1) - 1) * (data_matrix' * data_matrix);
        
        % calculate the principal components (pca_basis) and corresponding variances
        if frame == 1
            [pca_basis, pca_variances, ~] = svd(cov_matrix);
        else
            [pca_basis, pca_variances, ~] = svd((1 - compression_learning_rate) * old_cov_matrix + compression_learning_rate * cov_matrix);
        end
        
        % calculate the projection matrix as the first principal
        % components and extract their corresponding variances
        projection_matrix = pca_basis(:, 1:num_compressed_dim);
        projection_variances = pca_variances(1:num_compressed_dim, 1:num_compressed_dim);
        
        if frame == 1
            % initialize the old covariance matrix using the computed
            % projection matrix and variances
            old_cov_matrix = projection_matrix * projection_variances * projection_matrix';
        else
            % update the old covariance matrix using the computed
            % projection matrix and variances
            old_cov_matrix = (1 - compression_learning_rate) * old_cov_matrix + compression_learning_rate * (projection_matrix * projection_variances * projection_matrix');
        end
 %   end
    
    % project the features of the new appearance example using the new
    % projection matrix
    x = feature_projection(fo,xo_pca, projection_matrix, cos_window);
    
    % calculate the new classifier coefficients
    kf = fft2(dense_gauss_kernel(sigma, x));
    new_alphaf_num = yf .* kf;
    new_alphaf_den = kf .* (kf + lambda);
    
    if frame == 1
        % first frame, train with a single image
        alphaf_num = new_alphaf_num;
        alphaf_den = new_alphaf_den;
    else
        % subsequent frames, update the model
        alphaf_num = (1 - learning_rate) * alphaf_num + learning_rate * new_alphaf_num;
        alphaf_den = (1 - learning_rate) * alphaf_den + learning_rate * new_alphaf_den;
    end
    
    %save position
    positions(frame,:) = [pos target_sz];
    
    time = time + toc;
    
    %visualization
    if visualization == 1
        rect_position = [pos([2,1]) - target_sz([2,1])/2, target_sz([2,1])];
        if frame == 1,  %first frame, create GUI
            figure('Number','off', 'Name',['Tracker - ' video_path]);
            im_handle = imshow(uint8(im), 'Border','tight', 'InitialMag', 100 + 100 * (length(im) < 500));
            rect_handle = rectangle('Position',rect_position, 'EdgeColor','g');
            text_handle = text(10, 10, int2str(frame));
            set(text_handle, 'color', [0 1 1]);
        else
            try  %subsequent frames, update GUI
                set(im_handle, 'CData', im)
                set(rect_handle, 'Position', rect_position)
                set(text_handle, 'string', int2str(frame));
            catch
                return
            end
        end
        
        drawnow
%         pause
    end
end

fps = num_frames/time;
end
