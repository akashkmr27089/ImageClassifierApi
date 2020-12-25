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