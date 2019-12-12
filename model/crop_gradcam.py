#
# Cloud for ML Final Project
# Sahar Siddiqui
# crop_gradcam.py
#

import cv2 as cv
import model.image as ig
import numpy as np
from PIL import Image
from model.alexnet import load
from model.maps import guided_gradcam
from util.map_util import save_gradient_images, convert_to_grayscale, preprocess_image

def crop_relevant(input_image_path, imagenet_class_number=219,
					gradcam_output_file=None,cropped_output_file=None,unguided_gradcam_output_file=None):
	
	cam, cam_gb = guided_gradcam(load(), input_image_path, imagenet_class_number, None, colored=True)
	grayscale_cam_gb = convert_to_grayscale(cam_gb)

	width = Image.open(input_image_path).width
	height = Image.open(input_image_path).height

	save_gradient_images(cam_gb, gradcam_output_file)
	save_gradient_images(cam, unguided_gradcam_output_file)

	# if gradcam_output_file is not None:
	# 	cam_gb.save(gradcam_output_file)

	grayscale_cam_gb = grayscale_cam_gb.squeeze(0)

	input_img = Image.open(input_image_path).convert('RGB')
	input_img = preprocess_image(input_img)

	b, c, w, h = input_img.shape
	print(input_img.shape)

	grayscale_cam_gb -= np.min(grayscale_cam_gb)
	grayscale_cam_gb /= np.max(grayscale_cam_gb)
	grayscale_cam_gb = cv.resize(grayscale_cam_gb, (w,h))

	TR =  0.3

	mask = grayscale_cam_gb > TR

	input_img = input_img.squeeze()
	if len(input_img.shape) > 2: 
		input_img = input_img.permute(1, 2, 0)

	input_img = input_img.detach().cpu().numpy()
	crop = input_img.copy()
	crop[mask == 0] = 0

	crop = cv.resize(crop, (width, height))

	save_gradient_images(crop, cropped_output_file)

	# if cropped_output_file is not None:
	# 	cv.imwrite(cropped_output_file, crop)

