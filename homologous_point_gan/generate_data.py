import numpy as np
from point_processing import add_gaussian, place_point
import matplotlib.pyplot as plt
import os


def save_single_sample(fixed_image, moving_image, fixed_point, out_point, output_dir):
    '''
    @param fixed_image: single-layer uint8 image (MRI)
    @param moving_image: single-layer unit8 image (histology) grayscale
    @param fixed_point: A point on the fixed image (row, col)
    @param out_point: The corresponding point on the moving image (row, col)
    @param output_dir: The directory to write results
    Saves:
        gan_in.npy (Three-layer float32 np file containing layer 0 = fixed image layer 1 = moving image layer 2 gaussian single-point on fixed)
        fixed.png (A png showing the fixed image with a gaussian circle over the fixed point)
        moving.png (A png showing the moving image with a gaussian over the moving point)
    '''

    os.makedirs(output_dir, exist_ok=True)

    # Save the GAN input
    complete = np.dstack((fixed_image[:, :, np.newaxis], moving_image[:, :, np.newaxis])).astype(np.float32)
    complete = place_point(complete, fixed_point, variance=25.0)
    np.save(output_dir + "gan_in.npy", complete)

    # Save fixed image with gaussian blur circle
    fixed_img_with_gaussian = ((complete[:, :, -1] * 255.0) + fixed_image)
    fixed_img_with_gaussian[fixed_img_with_gaussian > 255] = 255
    fixed_img_with_gaussian = fixed_img_with_gaussian.astype(np.uint8)
    plt.imsave(output_dir + 'fixed.png', fixed_img_with_gaussian, cmap='gray')

    # Save moving with point
    moving_with_point = place_point(moving_image[:, :, np.newaxis], out_point, variance=25.0)
    moving_img_with_gaussian = (moving_with_point[:, :, -1] * 255.0 + moving_image)
    moving_img_with_gaussian[moving_img_with_gaussian > 255] = 255
    moving_img_with_gaussian = moving_img_with_gaussian.astype(np.uint8)
    plt.imsave(output_dir + 'moving.png', moving_img_with_gaussian, cmap='gray')
