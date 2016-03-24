% Local Feature Stencil Code
% CS 4495 / 6476: Computer Vision, Georgia Tech
% Written by James Hays

% Returns a set of interest points for the input image

% 'image' can be grayscale or color, your choice.
% 'feature_width', in pixels, is the local feature width. It might be
%   useful in this function in order to (a) suppress boundary interest
%   points (where a feature wouldn't fit entirely in the image, anyway)
%   or(b) scale the image filters being used. Or you can ignore it.

% 'x' and 'y' are nx1 vectors of x and y coordinates of interest points.
% 'confidence' is an nx1 vector indicating the strength of the interest
%   point. You might use this later or not.
% 'scale' and 'orientation' are nx1 vectors indicating the scale and
%   orientation of each interest point. These are OPTIONAL. By default you
%   do not need to make scale and orientation invariant local features.
function [x, y, confidence, scale, orientation] = get_interest_points(image, feature_width)

% Implement the Harris corner detector (See Szeliski 4.1.1) to start with.
% You can create additional interest point detector functions (e.g. MSER)
% for extra credit.

% If you're finding spurious interest point detections near the boundaries,
% it is safe to simply suppress the gradients / corners near the edges of
% the image.

% The lecture slides and textbook are a bit vague on how to do the
% non-maximum suppression once you've thresholded the cornerness score.
% You are free to experiment. Here are some helpful functions:
%  BWLABEL and the newer BWCONNCOMP will find connected components in 
% thresholded binary image. You could, for instance, take the maximum value
% within each component.
%  COLFILT can be used to run a max() operator on each sliding window. You
% could use this to ensure that every interest point is at a local maximum
% of cornerness.

% Step 0: Preparation: Set the variables and Gaussian filters

small_Gaussian = fspecial('gaussian', 3 .^ 2, 1);               % small Gaussian
large_Gaussian = fspecial('gaussian', feature_width .^ 2, 2);   % large Gaussian
[Guassian_dx, Guassian_dy] = gradient(small_Gaussian);          % 1st derivitive of small_Gaussian
alpha = 0.04;                                                   % alpha is between 0.04 and 0.06
img_size = size(image);                                         % image size

% Step 1: Calculate the Ix, Iy, Ix2, Iy2, Ixy

Ix = imfilter(image, Guassian_dx);
Iy = imfilter(image, Guassian_dy);
Ix2 = imfilter(Ix .^ 2, large_Gaussian);
Iy2 = imfilter(Iy .^ 2, large_Gaussian);
Ixy = imfilter(Ix .* Iy, large_Gaussian);

% Step 2: Calculate the harris value, apply the border and the thredshold
% on the result

harris = Ix2 .* Iy2 - Ixy .^ 2 - alpha .* (Ix2 + Iy2) .* (Ix2 + Iy2);
border = zeros(img_size);
border(feature_width + 1 : end - feature_width, feature_width + 1 : end - feature_width) = 1;
harris = harris .* border;
threshold = mean2(harris);
thresholded = harris > threshold;

% Step 3: Find the connected components, pick the local maximum and record
% the results to x, y and confidence.

cc = bwconncomp(thresholded);
num = cc.NumObjects;
x = zeros(num, 1);
y = zeros(num, 1);
confidence = zeros(num, 1);
img_size = size(image);
for idx = 1 : num
    pixels = cc.PixelIdxList{idx};
    component = harris(pixels);
    [maxNum, IDX] = max(component);
    [y(idx), x(idx)] = ind2sub(img_size, pixels(IDX));
    confidence(idx) = maxNum;
end

% Step 4: Calculate the inverse of the scale
sum = 0;
for ii = 1: length(x)
    for jj = 2 : length(x)
        sum = sum + ((x(ii)-x(jj))^2 + (y(ii)-y(jj))^2)^(1/2);
    end
end
scale = (length(x) ^ 2) / sum;
end
