# src/evaluate.py
import numpy as np
import tensorflow as tf
from preprocessing import load_data

def evaluate_model(model_path, data_dir, target_size=(128, 128)):
    _, _, _, _, X_test, y_test = load_data(data_dir, target_size)
    model = tf.keras.models.load_model(model_path)
    results = model.evaluate(X_test, y_test)
    print(f'Test Loss: {results[0]}, Test Accuracy: {results[1]}')

if __name__ == '__main__':
    evaluate_model('stroke_detection_model.h5', 'data')
