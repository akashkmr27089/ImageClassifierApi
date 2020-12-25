import base64
import flask
from flask import request, jsonify
import cv2

# image = open('./image.png')
img = cv2.imread('./image.png')  
image_string = base64.b64encode(img.read())
print(image_string)