from flask import Flask
from flask_sock import Sock
from PIL import Image, ImageOps
import os
from io import BytesIO
import base64
from keras.models import load_model
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
app = Flask(__name__)
sock = Sock(app)

@sock.route("/")
def receiveData(ws):
    data = ws.receive()
    prefix = 'data:image/jpeg;base64,'
    cut = data[len(prefix):]
    im = Image.open(BytesIO(base64.b64decode(cut)))
    im.save('test.jpg')
    image = Image.open('test.jpg').convert('RGB')
    model = load_model('keras_Model.h5', compile=False)
    # Load the labels
    class_names = open('labels.txt', 'r').readlines()
    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1.
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    # Replace this with the path to your image
    image = Image.open('test.jpg').convert('RGB')
    #resize the image to a 224x224 with the same strategy as in TM2:
    #resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    #turn the image into a numpy array
    image_array = np.asarray(image)
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    # Load the image into the array
    data[0] = normalized_image_array
    # run the inference
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]
    print('Class:', class_name, end='')
    print('Confidence score:', confidence_score)
    if confidence_score > 0.8:
        ws.send(class_name)