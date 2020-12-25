import flask
from flask import request, jsonify
import cv2
import numpy as np
import io
from PIL import Image
import matplotlib.pyplot as plt

'''
Content of Postman: 
1. Post Request
2. Body 
3. formdata
4. Set Value to file
Select image 
'''

app = flask.Flask(__name__)
app.config["DEBUG"] = True

ImageData = None

@app.route("/showImage", methods=["GET"])
def plotFig():
    global ImageData
    print("Type of Image " + str(type(ImageData)))
    base64.b6
    return 'print Successful'

@app.route("/name", methods=["POST"])
def setName():
    print(request.files['Image'])
    data = request.files['Image']
    data.save('./rImage.png')
    doc = data.read()
    image = np.array(Image.open(io.BytesIO(doc))) 
    global ImageData
    ImageData = image.astype(np.float)
    return "nation"

@app.route("/name")
def printName():
    return "Successfully send"

app.run()