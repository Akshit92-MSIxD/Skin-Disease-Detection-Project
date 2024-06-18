import os
from flask import Flask, render_template, request
from PIL import Image
from keras.models import load_model
import numpy as np
import tensorflow as tf
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configuration for dataset folder
DATASET_FOLDER = 'Dataset'

# Mapping of short forms to full disease names
DISEASE_NAMES = {
    'bkl': 'Benign Keratosis-like Lesions (bkl)',
    'mel': 'Melanoma (mel)',
    'nv': 'Melanocytic Nevi (nv)'
}

@app.route('/', methods=['GET'])
def hello_world():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    if 'imagefile' not in request.files:
        return 'No file part'
    
    imagefile = request.files['imagefile']
    subfolder = request.form.get('subfolder')
    
    if imagefile.filename == '':
        return 'No selected file'
    
    if subfolder not in ['bkl', 'mel', 'nv']:
        return 'Invalid subfolder selected'
    
    filename = secure_filename(imagefile.filename)
    file_path = os.path.join(DATASET_FOLDER, subfolder, filename)
    
    # Ensure the image file is uploaded and exists in the specified subfolder
    if not os.path.exists(file_path):
        return 'The selected file does not exist in the specified subfolder.'

    # Load the image directly from the uploaded file
    img = Image.open(imagefile)
    image_width, image_height = 224, 224
    img = img.resize((image_width, image_height))  # Resize the image to the desired dimensions
    img_array = np.array(img)  # Convert PIL image to NumPy array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize pixel values
    
    vgg16_model = load_model('VGG16/Saved Models/Vgg16_kfold_Three-class_82.h5')

    # # Define F1 Score metric
    # def f1_score(y_true, y_pred):
    #     true_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)))
    #     possible_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true, 0, 1)))
    #     predicted_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_pred, 0, 1)))

    #     precision = true_positives / (predicted_positives + tf.keras.backend.epsilon())
    #     recall = true_positives / (possible_positives + tf.keras.backend.epsilon())

    #     return 2 * ((precision * recall) / (precision + recall + tf.keras.backend.epsilon()))

    # inceptionV3_model = load_model('InceptionV3/Saved Models/Trained_model_Inception_three-class_76.h5', custom_objects={'f1_score': f1_score, 'precision': tf.keras.metrics.Precision(), 'recall': tf.keras.metrics.Recall()})
    # custom_model = load_model('CustomCNN/Saved Models/CustomModel_74.hdf5')

    result1 = vgg16_model.predict(img_array)
    # result2 = inceptionV3_model.predict(img_array)
    # result3 = custom_model.predict(img_array)
    
    # Assuming the order of classes is 'bkl', 'mel', 'nv'
    classes = ['bkl', 'mel', 'nv']
    prediction1 = classes[np.argmax(result1)]
    # prediction2 = classes[np.argmax(result2)]
    # prediction3 = classes[np.argmax(result3)]

    classification1 = '%s (%.2f%%)' % (DISEASE_NAMES[prediction1], result1[0][np.argmax(result1)] * 100)
    # classification2 = '%s (%.2f%%)' % (DISEASE_NAMES[prediction2], result2[0][np.argmax(result2)] * 100)
    # classification3 = '%s (%.2f%%)' % (DISEASE_NAMES[prediction3], result3[0][np.argmax(result3)] * 100)
    
    return render_template('index.html', prediction1=classification1)

if __name__ == '__main__':
    app.run(port=3000, debug=True)
