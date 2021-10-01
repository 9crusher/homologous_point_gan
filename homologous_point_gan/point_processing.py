# Standards
# Point coordinates are represented as (row, column) pairs
# All images are numpy arrays


def place_points(img, points, diameter=5):
    '''
    @param img: Numpy array containing an N layer image
    @param points: A list containing tuples or lists represnting point coordinates
    @param diameter: the diameter of the Gaussian circle

    Given an image and a list of points. Create a layer with points represented as 2D Gaussian noise circle
    centered around every point (row, col). The new layer is appended to the end of the input image.

    returns: New image with N + 1 layers. The final layer contains points represented as Gaussian noise circle
    '''
    height = img.shape[0]
    width = img.shape[1]

    for point in points:




def place_point():
    pass

def extract_point_centers(img):
    pass
