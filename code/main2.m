clear all;
clc;
image = imread('../data/images.jpg');
feature_width = 16;
[x, y, confidence] = get_interest_points(image, feature_width);
total_num = length(x);
feature_dimensionality = 128;
features = zeros(total_num, feature_dimensionality);

small_Gaussian = fspecial('gaussian', [feature_width, feature_width], 1);
large_Gaussian = fspecial('gaussian', [feature_width, feature_width], feature_width / 2);

[Guassian_dx, Guassian_dy] = gradient(small_Gaussian);

Ix = double(imfilter(image, Guassian_dx));
Iy = double(imfilter(image, Guassian_dy));

magnitude = sqrt(Ix .^ 2 + Iy .^ 2);
direction = mod(round((atan2(Iy, Ix) + 2 * pi) / (pi / 4)), 8);

grid_size = feature_width / 4;

% Loop over the (x,y) pairs
for point = 1 : total_num
    % Get the center of the grid
    xc = x(point);
    yc = y(point);
    % Apply the offset on the original (x,y) pair
    xs = xc - grid_size * 2;
    ys = yc - grid_size * 2;
    % Loop over each unit of the grid
    for xi = 0 : (grid_size - 1)
        for yi = 0 : (grid_size - 1)
            dir_grid = direction((xs + grid_size * xi + 1) : (xs + grid_size * xi + grid_size), ...
                (ys + grid_size * yi + 1) : (ys + grid_size * yi + grid_size));
            mag_grid = magnitude((xs + grid_size * xi + 1) : (xs + grid_size * xi + grid_size), ...
                (ys + grid_size * yi + 1) : (ys + grid_size * yi + grid_size));
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