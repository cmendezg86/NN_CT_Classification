#!/usr/bin/env python3

# src/predict.py
import tensorflow as tf
import cv2
import numpy as np
from preprocessing import load_and_preprocess_image

def predict_image(model_path, image_path, target_size=(128, 128)):
	# Cargar el modelo
	model = tf.keras.models.load_model(model_path)
	
	# Preprocesar la imagen
	image = load_and_preprocess_image(image_path, target_size)
	image = np.expand_dims(image, axis=0)  # Añadir una dimensión para el batch
	
	# Realizar la predicción
	prediction = model.predict(image)
	
	# Interpretar la predicción
	if prediction[0][0] > 0.5:
		label = 'ischaemic'
	else:
		label = 'hemorrhagic'
		
	return label, prediction[0][0]

if __name__ == '__main__':
	import sys
	if len(sys.argv) != 3:
		print("Usage: python predict.py <model_path> <image_path>")
	else:
		model_path = sys.argv[1]
		image_path = sys.argv[2]
		label, confidence = predict_image(model_path, image_path)
		print(f'The image is classified as: {label} with a confidence of {confidence:.4f}')
		