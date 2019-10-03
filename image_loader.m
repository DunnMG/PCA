clear all;
close all;

%% Loading Images
% Changes to sub-folder and loads in all images of .pgm type. Converts all
% images into a matrix of n rows by height x width columns.

cd Pics

h = 64;
w = 64;
d = h*w;

imagefiles = dir('*.pgm');
nfiles = length(imagefiles);    % Number of files found
images = zeros((h*w), nfiles, 1);
for i=1:nfiles
   currentfilename = imagefiles(i).name;
   currentimage = imread(currentfilename);
   images(:,i) = reshape(currentimage, 1, (h*w));
end

cd ..

%%
h = 64;
w = 64;

d = h*w;
% vectorize images
x = double(images);
%subtract mean
mean_matrix = mean(x,2);

x = bsxfun(@minus, x, mean_matrix);
% calculate covariance
s = cov(x');
% obtain eigenvalue & eigenvector
[V,D] = eig(s);
eigval = diag(D);
% sort eigenvalues in descending order
eigval = eigval(end:-1:1);
V = fliplr(V);


%%
% show mean and 1th through 15th principal eigenvectors
figure,subplot(4,4,1)
imagesc(reshape(mean_matrix, [h,w]))
title("Principal Components")
colormap gray
axis equal
xlim = 64;
ylim = 64;
for i = 1:15
    subplot(4,4,i+1)
    imagesc(reshape(V(:,i),h,w))
    axis equal
xlim = 64;
ylim = 64;
end

%%
% evaluate the number of principal components needed to represent 95% Total variance.
eigsum = sum(eigval);
csum = 0;
for i = 1:d
    csum = csum + eigval(i);
    tv = csum/eigsum;
    if tv > 0.95
        k95 = i;
        break
    end
end

%% Build an image
