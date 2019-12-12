#
# Cloud for ML Final Project
# Cole Smith
# alexnet.py
#

import torch
from torchvision import models

from torchvision import transforms


def load():
    """
    Loads a pre-trained AlexNet model

    :return: PyTorch AlexNet Model
    """
    return models.alexnet(pretrained=True)


def predict(img, model):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )])

    # Load and transform the image path given as arg
    # img = Image.open(img_path)
    img = torch.unsqueeze(transform(img), 0)
    model.eval()

    # Present the predicted classes for the image
    with open('imagenet_classes.txt') as f:
        out = model(img)
        labels = [line.strip() for line in f.readlines()]
        percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100

        _, indices = torch.sort(out, descending=True)
        res = [(labels[idx], percentage[idx].item(), int(idx)) for idx in indices[0][:5]]

        return res


# TODO
def retrain(images_dir, pretrained_prior=False):
    """
    Given a directory path that contains an equivalent
    structure to ImageNet, train a new instance of
    AlexNet on these images. The starting weights
    can be random, or a pretrained prior.

    :param images_dir:          Directory to training images in equivalent structure
                                to ImageNet
    :param pretrained_prior:    Whether or not to start with a pre-trained AlexNet model
    :return:
    """
    pass
