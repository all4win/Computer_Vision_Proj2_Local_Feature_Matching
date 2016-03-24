clear all;
clc;

% Step 0: to declare varibles
sigma = 1;
radius = 1;
box_size = 3;
threshold = 185;
small_Gaussian = fspecial('gaussian', 5 * sigma, sigma);
[Guassian_dx, Guassian_dy] = gradient(small_Gaussian);
large_Gaussian = fspecial('gaussian', 7 * sigma, sigma);
alpha = 0.06;

image = imread('../data/Notre Dame/921919841_a30df938f2_o.jpg'); % read the image
[height, width, layers] = size(image);
image = rgb2gray(image);
img_size = size(image);

% Step 1: calculate Ix, Iy, Ix2, Iy2, Ixy
% for i = 1 : 1
%     Ix(:,:,i) = imfilter(image(:,:,i), gaussianDx);
%     Iy(:,:,i) = imfilter(image(:,:,i), gaussianDy);
%     gIx2(:,:,i) = imfilter(Ix(:,:,i) .* Ix(:,:,i), lgaussianD);
%     gIy2(:,:,i) = imfilter(Iy(:,:,i) .* Iy(:,:,i), lgaussianD);
%     gIxIy(:,:,i) = imfilter(Ix(:,:,i) .* Iy(:,:,i), lgaussianD);
i = 1;
    Ix(:,:,i) = imfilter(image(:,:,i), Guassian_dx);
    Iy(:,:,i) = imfilter(image(:,:,i), Guassian_dy);
    Ix2(:,:,i) = imfilter(Ix(:,:,i) .^ 2, large_Gaussian);
    Iy2(:,:,i) = imfilter(Iy(:,:,i) .^ 2, large_Gaussian);
    Ixy(:,:,i) = imfilter(Ix(:,:,i) .* Iy(:,:,i), large_Gaussian);
% end

% Step 2: calculate the harris value and applied thredshold on local
% maximums.
harris = Ix2 .* Iy2 - Ixy .^ 2 - alpha .* (Ix2 + Iy2) .* (Ix2 + Iy2);

% mx_harris = ordfilt2(harris, box_size .^ 2, ones(box_size));
thresholded = harris > threshold;

% feature_width = 16;
% border = zeros(img_size);
% border(ceil(feature_width ./ 2) + 1 : end - ceil(feature_width ./ 2), ...
%     ceil(feature_width ./ 2) + 1 : end - ceil(feature_width ./ 2)) = 1;
% thresholded = thresholded & border;
imshow(thresholded);

% mask = thresholded(:,:,1) | thresholded(:,:,2) | thresholded(:,:,3);
mask = thresholded;
cc = bwconncomp(mask);
num = cc.NumObjects;
x = zeros(1, num);
y = zeros(1, num);
confidence = zeros(1, num);

for idx = 1 : num
    pixels = cc.PixelIdxList{idx};
    component = harris(pixels);
    avg = mean(component);
    [maxNum, IDX] = max(component);
    [x(idx), y(idx)] = ind2sub(img_size, pixels(IDX));
end
% figure; imshow(image); hold on,
% plot (y, x, 'ys');
% numOfPixels = cellfun(@numel, cc.PixelIdxList);
% sortedPixels = sort(numOfPixels, 'descend');
% figure;imshow(mask);
