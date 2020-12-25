import requests
from requests.exceptions import HTTPError

url = 'http://127.0.0.1:5000/showImage'

try:
    response = requests.get(url)