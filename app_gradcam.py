#
# Cloud for ML Final Project
# Sahar Siddiqui
# app_gradcam.py
#

import base64
import os
from io import BytesIO

import load_model_gradcam as load_model
from PIL import Image
from flask import Flask, request, jsonify

# ------------------------------------------------------------------
# FLASK APP
# ------------------------------------------------------------------

# Make dirs for ingress and egress
if not os.path.exists("tmp"):
    os.makedirs("tmp")

if not os.path.exists("output"):
    os.makedirs("output")

app = Flask(__name__)


@app.route('/', methods=["POST"])
def upload_image():
    """
    Returns JSON in the form:
        {
        'class':            str(cls),
        'probability':      float(conf),
        'gradCAM':          str(encoded_img),
        'cropped':          str(encoded_cropped_img),
        'gradCAM_unguided': str(encoded_unguided_img)
        }

    Where images are in base64.

    @return: JSON
    """
    data = request.get_json()

    # Handle empty request
    if data is None:
        print("[ ERR ] Request body invalid: empty")
        return "Error: Missing Request"

    # Decode image and predict output
    img = Image.open(BytesIO(base64.urlsafe_b64decode(data['img'])))
    print("[ INF ] Success: Image Loaded")

    return jsonify(load_model.predict(img))


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
