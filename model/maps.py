#
# Cloud for ML Final Project
# Cole Smith
# maps.py
#

from PIL import Image

from util.gradcam import GradCam
from util.guided_backprop import GuidedBackprop
from util.guided_gradcam import guided_grad_cam
from util.map_util import get_positive_negative_saliency, \
    save_gradient_images, preprocess_image, convert_to_grayscale, \
    get_positive_negative_saliency_IMPROVED


#
# Utility Functions
#

def _setup(model, img_path, target_class):
    """
    Returns the preprocessed target image and
    gradient for use in the maps.

    :param model:         A PyTorch model to use for gradients
    :param img_path:      Path to input image
    :param target_class:  Class of the input image (more accurately,
                          the class to consider for saliency)
    :return: img, grad
    """
    # Load and preprocess input image
    input_img = Image.open(img_path).convert('RGB')
    input_img = preprocess_image(input_img)

    # Use guided back propagation to generate the gradients
    gbp = GuidedBackprop(model)

    # Get gradients
    grad = gbp.generate_gradients(input_img, target_class)

    return input_img, grad


#
# Map Functions
#

def saliency_map(model, img_path, target_class, output_path, positive_saliency=True, improved=False):
    """
    Generates the saliency map of the given image
    at `img_path` using the trained `model`. The
    image is calculated using the gradients from
    this pre-trained model.

    The saliency map is saved to the output location.
    To forgo saving the image, set `output_path` to
    `None`.

    :param model:             A PyTorch model to use for gradients
    :param img_path:          Path to input image
    :param target_class       Class of the input image (more accurately,
                              the class to consider for saliency)
    :param output_path        Where to save the output image
    :param positive_saliency  Positive or Negative Saliency Map
    :return: Saliency map as PIL image object
    """

    input_img, grad = _setup(model, img_path, target_class)

    if improved:
        pos, neg = get_positive_negative_saliency_IMPROVED(grad)
    else:
        pos, neg = get_positive_negative_saliency(grad)

    # if output_path is not None:
    if positive_saliency:
        return save_gradient_images(pos, output_path)
    else:
        return save_gradient_images(neg, output_path)


def gradient_map(model, img_path, target_class, output_path, colored=True):
    """
    Generates the gradient map of the given image
    at `img_path` using the trained `model`. The
    image is calculated using the gradients from
    this pre-trained model.

    The gradient map is saved to the output location.
    To forgo saving the image, set `output_path` to
    `None`.

    :param model:             A PyTorch model to use for gradients
    :param img_path:          Path to input image
    :param target_class       Class of the input image (more accurately,
                              the class to consider for saliency)
    :param output_path        Where to save the output image
    :param colored:           Colored or grayscale gradient map
    :return: `None`
    """
    input_img, grad = _setup(model, img_path, target_class)

    if output_path is not None:
        if colored:
            return save_gradient_images(grad, output_path)
        else:
            grayscale = convert_to_grayscale(grad)
            return save_gradient_images(grayscale, output_path)


def guided_gradcam(model, img_path, target_class, output_path, colored=True):
    """
    Generates the guided gradient CAM of the given image
    at `img_path` using the trained `model`. The
    image is calculated using the gradients from
    this pre-trained model.

    The gradient CAM is saved to the output location.
    To forgo saving the image, set `output_path` to
    `None`.

    :param model:             A PyTorch model to use for gradients
    :param img_path:          Path to input image
    :param target_class       Class of the input image (more accurately,
                              the class to consider for saliency)
    :param output_path        Where to save the output image
    :param colored:           Colored or grayscale gradient map
    :return: `None`
    """
    input_img, grad = _setup(model, img_path, target_class)

    # Grad cam
    gcv2 = GradCam(model, target_layer=11)
    cam = gcv2.generate_cam(input_img, target_class)

    # Guided Grad cam
    cam_gb = guided_grad_cam(cam, grad)
    grayscale_cam_gb = convert_to_grayscale(cam_gb)

    return cam, cam_gb
    # if colored:
    #     return save_gradient_images(cam_gb, output_path)
    # else:
    #     return save_gradient_images(grayscale_cam_gb, output_path)
