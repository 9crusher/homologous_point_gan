# Standards
# Point coordinates are represented as (row, column) pairs
# All images are numpy arrays
from scipy.stats import multivariate_normal


def add_gaussian(layer, point, variance=1.0):
    '''
    @param layer: Layer on which to apply a 2D gaussian noise circle
    @param point: The coordinates of the center of the noise circle (row, col)
    @param variance: The variance of the noise circle

    returns: Existing layer contents + gaussian noise circle centered at point
    referenced: https://github.com/adgilbert/pseudo-image-extraction
    '''
    input_shape = layer.shape
    row_indices, col_indices = np.mgrid[0: input_shape[0], 0: input_shape[1]]
    all_coords = np.column_stack([row_indices.flatten(), col_indices.flatten()])
    cov = np.array([[variance, 0], [0, variance]], dtype=np.float32) 
    return layer + multivariate_normal.pdf(all_coords, mean=point, cov=cov).reshape(input_shape)

def extract_gaussian_centers(layer, thresh=0.05, window_size=5):
    '''
    @param layer: A 2D numpy array containing a heatmap

    returns: The coordinates of the centers of the Gaussian circles
    process adapted from GAN SRAF: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9094711
    '''
    height = layer.shape[0]
    width = layer.shape[1]
    point_coords = []
    layer[layer < thresh] = 0.0 # Eliminate noise

    for i in range(height - window_size + 1):
        for j in range(width - window_size + 1):
            



    


def place_point(img, point):
    '''
    @param img: Numpy array containing an N layer image
    @param points: A tuple or list represnting point coordinates

    Given an image and a list of points. Create a layer with points represented as 2D Gaussian noise circle
    centered around every point (row, col). The new layer is appended to the end of the input image.

    returns: New image with N + 1 layers. The final layer contains points represented as Gaussian noise circle
    '''
    height = img.shape[0]
    width = img.shape[1]
    assert point[0] < height, "Point row coordinate is not on the image"
    assert point[1] < width, "Point col coordinate is not on the image"

    heatmap_layer = add_gaussian(np.zeros((height, width)), point, variance=1.0)
    return np.dstack((img, heatmap_layer))


def extract_point_center(img, )

