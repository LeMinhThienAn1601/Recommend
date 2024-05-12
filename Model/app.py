from flask import Flask, request, jsonify
import uuid
from os import path, remove
from image_model import recommend_image_output
from flask_cors import CORS, cross_origin

app = Flask(__name__)

cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/', methods=['POST'])
@cross_origin()
def hello():
    file = request.files['file']

    filename = f'{str(uuid.uuid4())}.{file.filename.split(".")[-1]}'

    file_path = path.join('./archive/images', filename)

    file.save(file_path)

    recommend_images = recommend_image_output(filename)

    remove(file_path)

    return jsonify(
        success=True,
        images=recommend_images
    )