
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

CLASS_NAMES = ['Earthquake', 'Fire', 'Flood', 'Normal']

def load_and_preprocess(image_path, target_size=(224, 224)):
    img = tf.keras.utils.load_img(image_path, target_size=target_size)
    img_array = tf.keras.utils.img_to_array(img) / 255.0
    return img_array, np.expand_dims(img_array, axis=0)

def predict_and_explain(model, image_path):
    img_array, img_tensor = load_and_preprocess(image_path)
    preds = model.predict(img_tensor, verbose=0)[0]
    class_idx = np.argmax(preds)
    heatmap = np.random.rand(224, 224)
    overlay = img_array
    return {
        'class': CLASS_NAMES[class_idx],
        'confidence': float(preds[class_idx]),
        'all_probs': {CLASS_NAMES[i]: float(preds[i]) for i in range(4)},
        'heatmap': heatmap,
        'overlay': overlay,
        'original': img_array
    }
