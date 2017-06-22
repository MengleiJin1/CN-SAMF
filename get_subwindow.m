function  [fo ,out_pca]= get_subwindow(patch, w2c,hog_orientations,cell_size)

% [out_npca, out_pca] = get_subwindow(im, pos, sz, non_pca_features, pca_features, w2c)
%
% Extracts the non-PCA and PCA features from image im at position pos and
% window size sz. The features are given in non_pca_features and
% pca_features. out_npca is the window of non-PCA features and out_pca is
% the PCA-features reshaped to [prod(sz) num_pca_feature_dim]. w2c is the
% Color Names matrix if used.
   %ÐÂ¸ÄÁË
    fo= double(fhog(single(patch) / 255, cell_size, hog_orientations));
	fo(:,:,end) = [];  %remove all-zeros channel ("truncation feature")
	ss = size(fo);
    reim= imresize(patch, [ss(1) ss(2)]);
    out_npca = get_feature_map(reim,'gray', w2c);
    out_pca1 = get_feature_map(reim, 'cn', w2c);
    out_pca=cat(3,out_pca1,out_npca);
    fo= reshape(fo, [ss(1)*ss(2), size(fo, 3)]);
   % out_pca = reshape(out_pca1, [ss(1)*ss(2), size(out_pca1, 3)]);

end

