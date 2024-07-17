# src/main.py
import train
import evaluate

if __name__ == '__main__':
    # Entrenar el modelo
    train.train_model('data')
    
    # Evaluar el modelo
    evaluate.evaluate_model('stroke_detection_model.h5', 'data')
