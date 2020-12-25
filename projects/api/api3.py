import flask
from flask import request, jsonify

'''
Content of Postman: 
1. Post Request
2. Body 
3. raw 

{
    "data":"API is fun"
}
'''

app = flask.Flask(__name__)
app.config["DEBUG"] = True

@app.route("/name", methods=["POST"])
def setName():
    if request.method=='POST':
        posted_data = request.get_json()
        data = posted_data['data']
        return jsonify(str("Successfully stored  " + str(data)))

@app.route("/name", methods=["GET"])
def setName2():
    nation = {"alpha": 2}
    if request.method=='GET':
        return jsonify(nation)
        
app.run()