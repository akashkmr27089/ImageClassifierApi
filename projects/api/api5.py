import flask
from flask import request, jsonify
import cv2
import numpy as np
import io
from PIL import Image
import matplotlib.pyplot as plt
import base64

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

CommonData = None
CommonData2 = None

@app.route("/showImage", methods=["GET"])
def plotFig():
    global CommonData2
    # print(CommonData2)
    print(" The type of Commondata 2 :" + str(type(CommonData2)))
    # StringToBytes = bytes(CommonData2,  'utf-8') 
    decoded = base64.b64decode(CommonData2)
    img = np.array(Image.open(io.BytesIO(decoded)))
    print(img)
    return "Ok"

@app.route("/getImage", methods=["GET"])
def getImage():
    global CommonData 
    StringToBytes = bytes(CommonData,  'utf-8') 
    decoded = base64.b64decode(CommonData2)
    try:
        img = np.array(Image.open(io.BytesIO(decoded)))
    except:
        print(ex)
    print(img)
    # print(type(decoded))
    # print(decoded)
    # print(StringToBytes)
    return 'ok'


'''
Client Side Code :

import requests 
import base64
import numpy as np
import io
from PIL import Image

response = requests.get("http://127.0.0.1:5000/getImage")
jsonFile = response.json()
ImageString = jsonFile["ImageData"]   //ImageData in bytes 
ImageBytes = bytes(ImageString, 'utf-8')
decoded = base64.b64decode(ImageBytes[1:])
img = np.array(Image.open(io.BytesIO(decoded)))
im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

'''

@app.route("/displayImage", methods=["GET"])
def DisplayImage():
    global CommonData 
    return jsonify({'ImageData' : CommonData})
    

@app.route("/imageUpload", methods=["POST"])
def mask_image():
    fileData = request.files['Image'].read()
    npimg = np.frombuffer(fileData, np.uint8)
    img = cv2.imdecode(npimg,cv2.IMREAD_COLOR)
    ################ Image Processing ##############

    #################Converting it back##############
    img = Image.fromarray(img.astype("uint8"))
    rawBytes = io.BytesIO()
    img.save(rawBytes, "PNG")
    rawBytes.seek(0)
    print(rawBytes.read())
    rawBytes.seek(0)
    im_base64 = base64.b64encode(rawBytes.read())
    global CommonData, CommonData2
    # CommonData = im_base64
    CommonData = str(im_base64)
    CommonData2 = im_base64
    return jsonify({'ImageData' : str(im_base64)})


@app.route("/name")
def printName():
    return "Successfully send"

app.run()