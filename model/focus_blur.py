#
# Cloud for ML Final Project
# Cole Smith
# focus_blur.py
#

import cv2 as cv
import model.image as ig
import numpy as np
from PIL import Image
from model.alexnet import load
from model.maps import saliency_map
from util.map_util import save_gradient_images


def focus_blur(input_image_path, imagenet_class_number=219, output_file=None,
               mask_output_file=None, saliency_map_output_file=None):
    """
    Intelligently blurs the background of an input image using a combined positive / negative
    saliency map. This map undergoes luminosity filtering and Gaussian blurring to get the final
    greyscale mask of the foreground of the input image. This mask is then used to blur all but the
    salient part of the image for the given input class.

    @param input_image_path:            Path to the input image
    @param imagenet_class_number:       Class number from `data/imagenet_classes.txt` for input image
    @param output_file:                 Path to the blurred output image
    @param mask_output_file:            Path to the blurred output mask image
    @param saliency_map_output_file     Path to the unblurred mask image
    @return:                            PIL Image of blurred final image
    """
    p = saliency_map(load(), input_image_path, imagenet_class_number, None, positive_saliency=True)
    n = saliency_map(load(), input_image_path, imagenet_class_number, None, positive_saliency=False)
    combined = ig.add(p, n, mix=0.5)

    # Convert to OpenCV format
    grey = cv.cvtColor(np.array(combined), cv.COLOR_RGB2GRAY)

    # Get original image width and height
    w = Image.open(input_image_path).width
    h = Image.open(input_image_path).height

    # Resize salient mask to image size
    grey = cv.resize(grey, (w, h))

    if saliency_map_output_file is not None:
        cv.imwrite(saliency_map_output_file, grey)

    # Binary luminosity threshold on salient mask
    # Blur the mask, and convert it to RGB to use it with RGB images
    ret, mask = cv.threshold(grey, 10, 255, cv.THRESH_BINARY)
    mask = cv.GaussianBlur(mask, (51, 51), 11)
    mask = cv.cvtColor(mask, cv.COLOR_GRAY2RGB)

    if mask_output_file is not None:
        cv.imwrite(mask_output_file, mask)

    # Create the original and blurred image
    original = cv.imread(input_image_path)
    blurred_image = cv.GaussianBlur(original, (5, 5), 5)

    # Convert uint8 to float
    original = original.astype(float)
    blurred_image = blurred_image.astype(float)

    # Normalize the alpha mask to keep intensity between 0 and 1
    mask = mask.astype(float) / 255

    # Multiply the mask with the original
    original = cv.multiply(mask, original)

    # Multiply the blurred image (background) with ( 1 - mask )
    blurred_image = cv.multiply(1.0 - mask, blurred_image)

    result = cv.add(original, blurred_image)

    if output_file is not None:
        cv.imwrite(output_file, result)

    return save_gradient_images(result, None)
