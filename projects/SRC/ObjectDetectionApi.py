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
        print(max(predicted[0]), min(predicted[0]))
        predicted[0] = predicted[0]/(max(predicted[0]) - min(predicted[0]))
        topResult = list(predicted.argsort()[0][-5:])
        # print([int(x) for x in topResult])
        # print([np.float(predicted[0][int(x)]) for x in topResult])
        return jsonify({'ImageCategory' : str([ '{} : {}'.format(Vgg16Labels[int(x)], np.float(predicted[0][int(x)])) for x in topResult])})
    except:
        print("Some Error Occured")
        return "Error"  
    
app.run()