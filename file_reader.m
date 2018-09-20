clear;
clc;
%load('train_32x32.mat');
%%

img_size = 64;
num_channels = 3;
dirs_in_path = dir('./EuroSAT/');
X_img = zeros(img_size, img_size, num_channels, 0);
Y_lab = zeros(0, 1);
test_train_ratio = 0.8;

for i=3:12
    dir_path = strcat('./EuroSAT/', dirs_in_path(i).name, '/');
    files_in_dir = dir(dir_path);
    fprintf('Reading files in %s \n', dir_path);
    for j=3:size(files_in_dir, 1)
        file_path = strcat(dir_path, files_in_dir(j).name);
        img = imread(file_path);
        img = reshape(img, [img_size, img_size, num_channels, 1]);
        X_img = cat(4, X_img, img);
        Y_lab = cat(1, Y_lab, (i-3));
    end
end

num_examples = size(Y_lab, 1);
rand_idxs = randperm(num_examples);
X_img = X_img(:,:,:,rand_idxs);
Y_lab = Y_lab(rand_idxs, 1);

nrof_train = round(test_train_ratio*num_examples);
% train set
X = X_img(:,:,:,1:nrof_train);
Y = Y_lab(1:nrof_train,:);
save('train_64x64.mat', 'X', 'Y');

% test set
X = X_img(:,:,:,nrof_train+1:end);
Y = Y_lab(nrof_train+1:end,:);
save('test_64x64.mat', 'X', 'Y');

