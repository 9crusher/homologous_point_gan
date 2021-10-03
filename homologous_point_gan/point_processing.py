# Standards
# Point coordinates are represented as (row, column) pairs
# All images are numpy arrays
from scipy.stats import multivariate_normal
import numpy as np


def add_gaussian(layer, point, variance=1.0):
    '''
    @param layer: Layer on which to apply a 2D gaussian noise circle. Should be of type float
    @param point: The coordinates of the center of the noise circle (row, col)
    @param variance: The variance of the noise circle

    returns: Existing layer contents + gaussian noise circle centered at point.
             Values in heatmap are normalized between 0 and 1
    referenced: https://github.com/adgilbert/pseudo-image-extraction
    '''
    input_shape = layer.shape
    row_indices, col_indices = np.mgrid[0: input_shape[0], 0: input_shape[1]]
    all_coords = np.column_stack([row_indices.flatten(), col_indices.flatten()])
    cov = np.array([[variance, 0], [0, variance]], dtype=np.float32) 
    result = multivariate_normal.pdf(all_coords, mean=point, cov=cov).reshape(input_shape)
    normalized_result = result / np.amax(result)
    return layer + normalized_result


def extract_gaussian_centers(layer, activation_thresh=0.25, nnz_thresh=8, scan_radius=5):
    '''
    @param layer: A 2D numpy array containing a heatmap
    @param activation_thresh: Threshold below which values will be treated as noise
    @param scan_radius: The distance from the current pixel to analyze for the greatest value

    returns: The coordinates of the centers of the Gaussian circles
    process adapted from GAN SRAF: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9094711
    '''
    assert nnz_thresh <= 8 and nnz_thresh >= 0, "NNZ thresh must be in [0, 8]"

    height = layer.shape[0]
    width = layer.shape[1]
    point_coords = []
    layer[layer < activation_thresh] = 0.0 # Eliminate noise

    for i in range(scan_radius, height - scan_radius):
        for j in range(scan_radius, width - scan_radius):
            current_value = layer[i, j] 
            if current_value > 0:
                nnz_neighbors = np.sum((layer[i-1:i+2, j-1:j+2] > 0.0).astype(np.uint8)) - 1
                if nnz_neighbors >= nnz_thresh:
                    window = layer[i-scan_radius:i+scan_radius+1, j-scan_radius:j+scan_radius+1]
                    if np.amax(window) == current_value:
                        point_coords.append((i, j))
    return point_coords


def place_point(img, point, variance=1.0):
    '''
    @param img: Numpy array containing an N layer image
    @param points: A tuple or list represnting point coordinates
    @param variance: The variance of the gaussian circle

    Given an image and a list of points. Create a layer with points represented as 2D Gaussian noise circle
    centered around every point (row, col). The new layer is appended to the end of the input image.

    returns: New image with N + 1 layers. The final layer contains points represented as Gaussian noise circle
    '''
    height = img.shape[0]
    width = img.shape[1]
    assert point[0] < height, "Point row coordinate is not on the image"
    assert point[1] < width, "Point col coordinate is not on the image"

    heatmap_layer = add_gaussian(np.zeros((height, width), dtype=np.float32), point, variance=variance)
    return np.dstack((img, heatmap_layer))


def extract_point(img):
    '''
    @param: An image with a last layer containing a heatmap point

    returns: The coordinates of the point contained in the heatmap layer (row, col)
    '''
    points = extract_gaussian_centers(img[:, :, -1])
    assert len(points) == 1, "Found {0} points but expected 1".format(len(points))
    return points[0]
