% Local Feature Stencil Code
% CS 4495 / 6476: Computer Vision, Georgia Tech
% Written by James Hays

% Returns a set of feature descriptors for a given set of interest points. 

% 'image' can be grayscale or color, your choice.
% 'x' and 'y' are nx1 vectors of x and y coordinates of interest points.
%   The local features should be centered at x and y.
% 'feature_width', in pixels, is the local feature width. You can assume
%   that feature_width will be a multiple of 4 (i.e. every cell of your
%   local SIFT-like feature will have an integer width and height).
% If you want to detect and describe features at multiple scales or
% particular orientations you can add input arguments.

% 'features' is the array of computed features. It should have the
%   following size: [length(x) x feature dimensionality] (e.g. 128 for
%   standard SIFT)

function [features] = get_features(image, x, y, feature_width, scale)

% To start with, you might want to simply use normalized patches as your
% local feature. This is very simple to code and works OK. However, to get
% full credit you will need to implement the more effective SIFT descriptor
% (See Szeliski 4.1.2 or the original publications at
% http://www.cs.ubc.ca/~lowe/keypoints/)

% Your implementation does not need to exactly match the SIFT reference.
% Here are the key properties your (baseline) descriptor should have:
%  (1) a 4x4 grid of cells, each feature_width/4.
%  (2) each cell should have a histogram of the local distribution of
%    gradients in 8 orientations. Appending these histograms together will
%    give you 4x4 x 8 = 128 dimensions.
%  (3) Each feature should be normalized to unit length
%
% You do not need to perform the interpolation in which each gradient
% measurement contributes to multiple orientation bins in multiple cells
% As described in Szeliski, a single gradient measurement creates a
% weighted contribution to the 4 nearest cells and the 2 nearest
% orientation bins within each cell, for 8 total contributions. This type
% of interpolation probably will help, though.

% You do not have to explicitly compute the gradient orientation at each
% pixel (although you are free to do so). You can instead filter with
% oriented filters (e.g. a filter that responds to edges with a specific
% orientation). All of your SIFT-like feature can be constructed entirely
% from filtering fairly quickly in this way.

% You do not need to do the normalize -> threshold -> normalize again
% operation as detailed in Szeliski and the SIFT paper. It can help, though.

% Another simple trick which can help is to raise each element of the final
% feature vector to some power that is less than one.


% Allocate the space for features and set the variables
feature_width = round(feature_width  * scale / 4) * 4;
total_num = length(x);
feature_dimensionality = 128;
features = zeros(total_num, feature_dimensionality);

Gaussian = fspecial('gaussian', [feature_width, feature_width], 1);
smooth_Gaussian = fspecial('gaussian', [feature_width, feature_width], feature_width / 2);
[Guassian_dx, Guassian_dy] = gradient(Gaussian);

% Differentiate the image by appling the diffierentiated guassian filter
Ix = double(imfilter(image, Guassian_dx));
Iy = double(imfilter(image, Guassian_dy));

% Calculate the magnitude and direction of each pixel
magnitude = sqrt(Ix .^ 2 + Iy .^ 2);
direction = mod(round((atan2(Iy, Ix) + 2 * pi) / (pi / 4)), 8);

% Loop over the (x,y) pairs
for point = 1 : total_num
    % Get the center of the grid
    xc = x(point);
    yc = y(point);
    % Apply the offset on the original (x,y) pair
    grid_size = feature_width / 4;
    xs = xc - grid_size * 2;
    ys = yc - grid_size * 2;
    % Smooth the magitude matrix
    large_mag_grid = magnitude(ys: ys + feature_width - 1, xs: xs + feature_width - 1);
    large_dir_grid = direction(ys: ys + feature_width - 1, xs: xs + feature_width - 1);
    large_mag_grid = imfilter(large_mag_grid, smooth_Gaussian);
    % Loop over each unit of the grid
    for xi = 0 : 3
        for yi = 0 : 3
            dir_grid = large_dir_grid((grid_size * xi + 1) : (grid_size * xi + grid_size), ...
                (grid_size * yi + 1) : (grid_size * yi + grid_size));
            mag_grid = large_mag_grid((grid_size * xi + 1) : (grid_size * xi + grid_size), ...
                (grid_size * yi + 1) : (grid_size * yi + grid_size));
            for d = 0 : 7
                mask = dir_grid == d;
                features(point, (xi * 32 + yi * 8) + d + 1) = sum(sum(mag_grid(mask)));
            end
        end
    end
    % Normalize each feature
    mag_sum = sum(features(point, :));
    features(point, :) = features(point, :) / mag_sum;
end

end