# src/train.py
import numpy as np
from preprocessing import load_data
from model import create_model
import matplotlib.pyplot as plt

def train_model(data_dir, target_size=(128, 128), batch_size=32, epochs=20):
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(data_dir, target_size)
    model = create_model(input_shape=(target_size[0], target_size[1], 3))
    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val))
    
    # Guardar el modelo
    model.save('stroke_detection_model.h5')

    # Crear gráficos de pérdida y precisión
    plt.figure()
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.savefig('loss_plot.png')

    plt.figure()
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    plt.savefig('accuracy_plot.png')

if __name__ == '__main__':
    train_model('data')
