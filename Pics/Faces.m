%% Faces

% Specify number of images
%num_imgs = 7;
% Enter in the location of your images (relative to MATLAB script)

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

%%
% vectorize images
x = double(images);
x = x';
Xn = x;

num_imgs = nfiles;

%% Build matrix

Mi = 64;
Ni = 64;   % image resize parameter


% Xn = zeros(num_imgs, Mi * Ni);
% for i = 1:num_imgs
% % specify the filename - it is easier if they have the same name and are
% % numbered sequentially
% file_str = [dir_str 'Apple' num2str(i) '.jpg'];
% data = imread(file_str);
% img = rgb2gray(data);
% img = imresize(img, [Mi Ni]);
% % Display the image
% figure(i);
% imagesc(img);
% colormap('gray');
% title(['Gauss ' num2str(i)]);
% axis equal;
% set(gca, 'Visible', 'off');
% Xn(i, :) = reshape(img, [Mi*Ni],1);
% end


%Xn = transpose(Xn);     % added -MD

%% Code from Part A
% d = size(Xn,1);  % for a 1 dimensional matrix -MD
d = num_imgs;
Xn = double(Xn);    % convert integers to double

N = size(Xn, 2);

% Compute the average value of each variable
mu = mean(Xn,2);

%%% mean_image = reshape(Xn,[Mi,Ni]); - can clear

% Replicate the average vector into a matrix
mu_mat = repmat( mu,1,N);
% Subtract the average away from every column
Xhat = Xn - mu_mat;
% Preallocate the scaled matrix
% Xhats = zeros(size(Xhat));
% % Populate the scaled matrix by row
% vecSquare = zeros(size(Xhat));  % added for transpose method -MD
% for i=1:num_imgs
%     %Xhats(i,:) = normalize(Xhat(i,:));
%     for j=1:N
%         vecSquare(i,j) = Xhat(i,j).*Xhat(i,j);
%     end
%     sums = sum(vecSquare(i,:));
%     
%     Xhats(i,:) = Xhat(i,:) / sqrt(sums); 
% end
% % Compute the correlation matrix
% C = Xhats * Xhats';
% Compute the sample covariance matrix
    S = (Xhat*Xhat')/(N-1);
     %S = cov(Xhat');
% Compute the trace of S, i.e. the total variance
Trace_S = trace(S);
% Find the eigenvalues and eigenvectors of S
[Q,Dvec] = eig(S,'vector');
% Sort in descending order
[Dvec, perm] = sort(Dvec, 'descend');
Q = Q(:,perm);
% Determine the principal components
PComps = Q'*Xhat;

%% 
%PComps = ____; % see above -MD

pca_img = zeros([Mi, Ni ,num_imgs]);
for i = 1:num_imgs
pca_img(:,:,i) = reshape(PComps(i,:),Mi,Ni);
end

%%
mean_matrix = mean(Xn,1);

% show mean and 1th through 15th principal eigenvectors
figure,subplot(4,4,1)
imagesc(reshape(mean_matrix, [h,w]))
colormap gray
axis equal
xlim = 64;
ylim = 64;
for i = 1:15
    subplot(4,4,i+1)
    imagesc(pca_img(:,:,i));
    axis equal
xlim = 64;
ylim = 64;
end

%%
% which image to reconstruct
image_indx = 72;
% how many dimensions to recreate
basis_dim = 100;
imgproj=Q(image_indx,1)*pca_img(:,:,1);
for i=2:basis_dim
imgproj= imgproj+ Q(image_indx,i)*pca_img(:,:,i);
end
figure;
imagesc(imgproj);
title('REPLACE ME')
colormap('gray')
axis equal
set(gca, 'Visible', 'off');




