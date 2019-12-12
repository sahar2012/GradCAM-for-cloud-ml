#
# Cloud for ML Final Project
# Cole Smith
# image.py
# Provides image manipulation operations
#

import blend_modes
import numpy as np
from PIL import Image

from util.map_util import convert_to_grayscale


#
# Convert Functions
#

def _conv_to_rbga(i):
    if i.mode == "RGBA":
        return i
    return i.convert("RGBA")


def _conv_to_rgb(i):
    background = Image.new('RGB', i.size, (255, 255, 255))
    alpha_composite = Image.alpha_composite(background, i)
    return alpha_composite


def black_to_alpha(i):
    # pixdata = i.load()
    #
    # width, height = i.size
    # for y in range(height):
    #     for x in range(width):
    #         if pixdata[x, y] == (0, 0, 0, 255):
    #             pixdata[x, y] = (0, 0, 0, 0)
    x = np.asarray(i.convert('RGBA')).copy()
    x[:, :, 3] = (255 * (x[:, :, :3] == 0).any(axis=2)).astype(np.uint8)

    return to_pil(x)


def to_numeric(i):
    """
    Converts image `i` to a numeric
    representation.

    :param i: PIL Image
    :return:  Numpy array of floats
    """
    return np.array(i).astype(float)


def to_pil(i):
    """
    Converts numeric array to PIL Image.

    :param i: Numeric Array (Numpy)
    :return: PIL Image
    """
    return Image.fromarray(np.uint8(i))


def size_to_same(reference, target):
    """
    Resizes `target` to the size of `reference`
    and returns both.

    :param reference: PIL Image
    :param target:    PIL Image
    :return:          PIL Image, PIL Image
    """
    return reference, target.resize(reference.size)


#
# Loaders
#

def from_file(file_path):
    """
    Loads PIL Image from file.

    :param file_path: File path string to image
    :return: PIL Image
    """
    return Image.open(file_path)


#
# Conversion Functions
#

def greyscale(i):
    """
    Converts image to greyscale using
    function from `util`
    :param i: PIL Image
    :return: Black and White PIL Image
    """
    raw = to_numeric(i)
    g = convert_to_grayscale(raw)
    return Image.fromarray(np.uint8(g))


#
# Blending Functions
#

def multiply(foreground, background, mix=0.7):
    """
    Multiplies the foreground by the background.

    :param foreground:  Foreground PIL Image
    :param background:  Background PIL Image
    :param mix:         Amount of foreground that contributes
                        to the final image. Between [0.0, 1.0]
    :return:            Blended PIL Image
    """
    foreground = _conv_to_rbga(foreground)
    background = _conv_to_rbga(background)

    background, foreground = size_to_same(background, foreground)

    bkg = to_numeric(background)
    frg = to_numeric(foreground)

    raw = blend_modes.multiply(bkg, frg, mix)

    return to_pil(raw)


def add(foreground, background, mix=0.7):
    """
    Adds the foreground by the background.

    :param foreground:  Foreground PIL Image
    :param background:  Background PIL Image
    :param mix:         Amount of foreground that contributes
                        to the final image. Between [0.0, 1.0]
    :return:            Blended PIL Image
    """
    foreground = _conv_to_rbga(foreground)
    background = _conv_to_rbga(background)

    background, foreground = size_to_same(background, foreground)

    bkg = to_numeric(background)
    frg = to_numeric(foreground)
    raw = blend_modes.addition(bkg, frg, mix)

    return to_pil(raw)


def divide(foreground, background, mix=0.7):
    """
    Divides the foreground by the background.

    :param foreground:  Foreground PIL Image
    :param background:  Background PIL Image
    :param mix:         Amount of foreground that contributes
                        to the final image. Between [0.0, 1.0]
    :return:            Blended PIL Image
    """
    foreground = _conv_to_rbga(foreground)
    background = _conv_to_rbga(background)

    background, foreground = size_to_same(background, foreground)

    bkg = to_numeric(background)
    frg = to_numeric(foreground)
    raw = blend_modes.divide(bkg, frg, mix)

    return to_pil(raw)


def subtract(foreground, background, mix=0.7):
    """
    Subtracts the foreground by the background.

    :param foreground:  Foreground PIL Image
    :param background:  Background PIL Image
    :param mix:         Amount of foreground that contributes
                        to the final image. Between [0.0, 1.0]
    :return:            Blended PIL Image
    """
    foreground = _conv_to_rbga(foreground)
    background = _conv_to_rbga(background)

    background, foreground = size_to_same(background, foreground)

    bkg = to_numeric(background)
    frg = to_numeric(foreground)
    raw = blend_modes.subtract(bkg, frg, mix)

    return to_pil(raw)


def overlay(foreground, background, mix=0.7):
    """
    Overlays the foreground by the background.

    :param foreground:  Foreground PIL Image
    :param background:  Background PIL Image
    :param mix:         Amount of foreground that contributes
                        to the final image. Between [0.0, 1.0]
    :return:            Blended PIL Image
    """
    foreground = _conv_to_rbga(foreground)
    background = _conv_to_rbga(background)

    background, foreground = size_to_same(background, foreground)

    bkg = to_numeric(background)
    frg = to_numeric(foreground)
    raw = blend_modes.normal(bkg, frg, mix)

    return to_pil(raw)
