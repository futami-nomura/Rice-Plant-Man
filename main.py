from flask import Flask, redirect, request, jsonify
# from flask_scss import Scss
from keras import models
import numpy as np
from PIL import Image
import io
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
from keras import backend as K

app = Flask(__name__)
#Scss(app, static_dir='static', asset_dir='assets')
model = None


def load_model():
    global model
    model = models.load_model('sep_05.h5')
    model.summary()
    print('Loaded the model')


@app.route('/index')
def index():
    return render_template('/static/index.html')

@app.route('/')
def detail():
    return redirect('/static/detail.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.files and 'picfile' in request.files:
        img = request.files['picfile'].read()
        img = Image.open(io.BytesIO(img))
        img.save('test.jpg')
        img = np.asarray(img) / 255.
        img = np.expand_dims(img, axis=0)
            #K.clear_session()
        pred = model.predict(img)
            # pred = model.predict(img)
        disease = [
                'ごま葉枯病',
                'いもち病',
                '縞葉枯病',
                ]
        confidence = str(round(max(pred[0]), 3))
        pred = disease[np.argmax(pred)]
        data = dict(pred=pred, confidence=confidence)
        return jsonify(data)
    return 'Picture info did not get saved.'


@app.route('/currentimage', methods=['GET'])
def current_image():
    fileob = open('test.jpg', 'rb')
    data = fileob.read()
    return data


if __name__ == '__main__':
    K.clear_session()
    load_model()
    model._make_predict_function()
    app.run(debug=True,host="0.0.0.0",port=5000)
