import flask
from flask import request, jsonify
import cv2
import numpy as np
import io
from PIL import Image
import matplotlib.pyplot as plt
import base64

from torchvision import transforms, datasets, models
from PIL import Image
from torchsummary import summary
from ModelDict import Vgg16Labels

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
print(" Loading Model ")
model = models.vgg16(pretrained=True)
print(" Model Loaded ")

imageTransrom = transforms.Compose([
        transforms.CenterCrop(size=224),  # Image net standards
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

def ImageTransformation(data):
    global imageTransrom
    try:
        returnData = imageTransrom(data)
        return returnData.resize(1,3,224,224)
    except:
        print(" Problem in Image Transformation ")
    return None

@app.route("/imageCat", methods=["POST"])
def imageCat():
    global model
    fileData = request.files['Image'].read()
    # filename = request.files['Image'].filename
    npimg = np.frombuffer(fileData, np.uint8)
    img = cv2.imdecode(npimg,cv2.IMREAD_COLOR)
    img = Image.fromarray(img)  #Comverting image to PIL Type
    ################ Image Processing ##############
    try:
        transformedImage = ImageTransformation(img)
        predicted = model(transformedImage)
        return jsonify({'ImageCategory' : str(Vgg16Labels[int(predicted.argmax())])})
    except:
        print("Some Error Occured")
        return "Error"

    #################Converting it back##############
    

'''
Client Side Code :

import requests 
import base64
import numpy as np
import io
from PIL import Image
import cv2
import matplotlib.pyplot as plt

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
    filename = request.files['Image'].filename
    npimg = np.frombuffer(fileData, np.uint8)
    img = cv2.imdecode(npimg,cv2.IMREAD_COLOR)
    ################ Image Processing ##############

    #################Converting it back##############
    img = Image.fromarray(img.astype("uint8"))
    rawBytes = io.BytesIO()
    img.save(rawBytes, "PNG")
    rawBytes.seek(0)
    im_base64 = base64.b64encode(rawBytes.read())
    global CommonData, CommonData2
    # CommonData = im_base64
    CommonData = str(im_base64)
    return jsonify({'ImageData' : str(im_base64)})


@app.route("/name")
def printName():
    return "Successfully send"

if __name__ == "__main__":


    app.run()