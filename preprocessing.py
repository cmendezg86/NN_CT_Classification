# src/preprocessing.py
import os
import cv2
import numpy as np

def load_and_preprocess_image(image_path, target_size):
	image = cv2.imread(image_path)
	image = cv2.resize(image, target_size)
	image = image / 255.0  # Normalizar
	return image

def load_data(data_dir, target_size=(128, 128)):
	categories = ['Train', 'Validation', 'Test']
	class_names = ['hemorrhagic', 'ischaemic']
	data = {'Train': {'images': [], 'labels': []},
			'Validation': {'images': [], 'labels': []},
			'Test': {'images': [], 'labels': []}}
	
	for category in categories:
		for class_name in class_names:
			folder_path = os.path.join(data_dir, category, class_name)
			label = class_names.index(class_name)
			for filename in os.listdir(folder_path):
				if filename.endswith('.jpg') or filename.endswith('.png'):
					image_path = os.path.join(folder_path, filename)
					image = load_and_preprocess_image(image_path, target_size)
					data[category]['images'].append(image)
					data[category]['labels'].append(label)
					
	return (np.array(data['Train']['images']), np.array(data['Train']['labels']),
			np.array(data['Validation']['images']), np.array(data['Validation']['labels']),
			np.array(data['Test']['images']), np.array(data['Test']['labels']))
			
