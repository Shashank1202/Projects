from flask import render_template,Flask,flash, request, redirect
import requests,os,sys
from werkzeug.utils import secure_filename
from keras.models import load_model
from keras.preprocessing import image
from keras import backend as K
import numpy as np
UPLOAD_FOLDER = os.path.abspath('uploads')

# Creating Flask App
app = Flask(__name__)
app.secret_key = "123jmdfsahdagdja"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Bootstrap the app
app = Flask(__name__)

# Allowed Extensions
ALLOWED_EXTENSIONS = set(['jpg', 'jpeg'])

# Classes
CLASSES = ('Glaucoma','Non Glaucoma')

def predictions(filename):
    # Load model
    model = load_model(os.path.abspath('model2.h5'));
    
    # Converting Image to np array
    img_prd = image.load_img(os.path.abspath('uploads/'+filename),target_size=(150,150));
    img_prd = image.img_to_array(img_prd);
    img_pred = np.expand_dims(img_prd, axis=0);
    
    # Predicting the class 
    result = model.predict(img_pred);
    
    # Clearing the session
    K.clear_session()
    
    # Clearing the memory
    del model,img_pred,img_prd
    return result

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET','POST'])
def home(): 
    K.clear_session()
    if request.method == 'POST':
        if 'file' not in request.files:
            result =  "No file selected"
            return result
        file = request.files['file']
        if file.filename == '':
            result = "No file selected for uploading"
            return result
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(os.path.abspath('uploads'), filename))
            print('File successfully uploaded')
            result = predictions(filename)
            prd_id = int(result[0][0])
            if prd_id is 0:
                result = ""+ CLASSES[prd_id]
            else:
                result = ""+ CLASSES[prd_id]
            return result
            return render_template('upload.html',result = result)
        else:
            return 'Allowed file types are jpg, jpeg'
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(port=8080,threaded=True)