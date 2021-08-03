import os
from flask import Flask, render_template
from utils import *


app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template('home.html')

@app.route('/process', methods=['GET'])
def process():
    data = data_read()
    return data.to_json()

@app.route('/splint_data_maker', methods=['GET'])
def splint():
    data = data_read()
    return data.to_json()

if __name__ == '__main__':
    app.run(debug=True)