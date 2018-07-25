from flask import Flask, redirect, request, jsonify
from keras import models
import numpy as np
from PIL import Image
import io
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input


app = Flask(__name__)
model = None


def load_model():
    global model
    model = models.load_model('model.04-4.33.hdf5')
    model.summary()
    print('Loaded the model')


@app.route('/')
def index():
    return redirect('/static/index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.files and 'picfile' in request.files:
        img = request.files['picfile'].read()
        img = Image.open(io.BytesIO(img))
        img.save('test.jpg')

        x = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        """
        pred = model.predict(img)

        x = img.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        """

        y_pred = model.predict(img)

        if y_pred <0.5:
            confidence = str(y_pred[0][0])
            pred = "rice-plant"
        else:
            confidence = str(y_pred[0][0])
            pred = "dog"

        """
        players = [
            'rice-plant',
            'other',
        ]

        confidence = str(round(max(pred[0]), 3))
        pred = players[np.argmax(pred)]

        """
        data = dict(pred=pred, confidence=confidence)
        return jsonify(data)

    return 'Picture info did not get saved.'


@app.route('/currentimage', methods=['GET'])
def current_image():
    fileob = open('test.jpg', 'rb')
    data = fileob.read()
    return data


if __name__ == '__main__':
    load_model()
    # model._make_predict_function()
    app.run(debug=False, port=5000)
