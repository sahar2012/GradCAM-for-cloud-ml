#
# Cloud for ML Final Project
# Sahar Siddiqui
# load_model_gradcam.py
#

import base64
import os
import uuid

from model.alexnet import load
from model.alexnet import predict as predict_alexnet
from model.crop_gradcam import crop_relevant


# -----------------------------------------------------
# -- Inference
# -----------------------------------------------------

def predict(img):
    """
    Returns in JSON format the class, probability of class, the
    GradCAM output image, the cropped image

    @param img:
    @return:
    """
    # Write down to tmp file
    filename = str(uuid.uuid4()) + ".jpg"
    filepath = os.path.join("tmp", filename)
    img.save(filepath)

    # Infer class
    (cls, conf, idx) = sorted(predict_alexnet(img, load()), reverse=True)[0]

    # Write to output file
    output = os.path.join("output", filename)
    cropped_output = os.path.join("output", 'cropped_' + filename)
    output_unguided = os.path.join("output", 'unguided_gradcam_' + filename)

    crop_relevant(filepath, imagenet_class_number=idx,
                    gradcam_output_file=output,cropped_output_file=cropped_output,
                    unguided_gradcam_output_file=output_unguided)

    # Output file to base64
    with open(output, "rb") as fp:
        encoded_img = base64.b64encode(fp.read())

    # Cropped Output file to base64
    with open(cropped_output, "rb") as fp:
        encoded_cropped_img = base64.b64encode(fp.read())

    # Cropped Output file to base64
    with open(output_unguided, "rb") as fp:
        encoded_unguided_img = base64.b64encode(fp.read())

    # Remove tmp and output files
    os.remove(filepath)
    os.remove(output)
    os.remove(cropped_output)

    response = {'class': str(cls),
                'probability': float(conf),
                'gradCAM': encoded_img.decode('utf-8'),
                'cropped': encoded_cropped_img.decode('utf-8'),
                'gradCAM_unguided': encoded_unguided_img.decode('utf-8')
                }

    return response
