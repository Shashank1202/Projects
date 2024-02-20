from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import uuid
import tensorflow as tf
from PIL import Image
import numpy as np

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('tumor_detection_model.h5')

# Define the upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Define a route to render the HTML page
@app.route('/')
def index():
    return render_template('index.html')

# Define a route to handle file uploads and make predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Get the uploaded file from the request
    file = request.files['file']

    # Generate a unique filename
    filename = str(uuid.uuid4()) + '.jpg'

    # Save the uploaded file to the uploads folder
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

    # Preprocess the image
    img = Image.open(file.stream).convert('RGB')
    img = img.resize((224, 224))  # Adjust the size as per your model's input requirements
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # Make predictions
    predictions = model.predict(img_array)
    result = "Tumor" if predictions[0][0] > 0.5 else "Non-Tumor"

    # Return the result along with the filename
    return jsonify({'result': result, 'filename': filename})

# Define a route to serve uploaded images
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
