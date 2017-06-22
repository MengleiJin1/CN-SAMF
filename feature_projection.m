function z = feature_projection(fo,x_pca, projection_matrix, cos_window)

% z = feature_projection(x_npca, x_pca, projection_matrix, cos_window)
%
% Calculates the compressed feature map by mapping the PCA features with
% the projection matrix and concatinates this with the non-PCA features.
% The feature map is then windowed.
    [height, width] = size(cos_window);
    [num_pca_in, num_pca_out] = size(projection_matrix);
    
    % project the PCA-features using the projection matrix and reshape
    % to a window
    x_proj_pca = reshape(fo * projection_matrix, [height, width, num_pca_out]);
    z = cat(3, x_pca, x_proj_pca);
% do the windowing of the output
z = bsxfun(@times, cos_window, z);
end