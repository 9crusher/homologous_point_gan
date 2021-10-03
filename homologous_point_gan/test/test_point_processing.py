# To run, call 'pytest homologous_point_gan/test/ -rP' from root dir
from homologous_point_gan.point_processing import place_point, extract_point
import numpy as np
import pytest

def create_blank_image():
    """Returns a dummy RGB image (512x512)"""
    return np.zeros((512, 512, 3))


def test_point_placement_and_extraction():
    point_location = (41, 235)
    test_image = create_blank_image()
    placed_image = place_point(test_image, point_location)
    extracted_point = extract_point(placed_image)
    assert point_location == extracted_point, "Point coordinates do not match"


