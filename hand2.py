from flask import Flask, request, jsonify
from matplotlib import pyplot as plt
import numpy as np
import cv2
import base64
from rembg import remove, new_session

model_name = "u2net_human_seg"
session = new_session(model_name)

app = Flask(__name__)

@app.route('/remove_background', methods=['POST'])
def remove_background():
    # Get the image matrix from the request
    image_matrix = np.array(request.json['image'], dtype=np.uint8)
    image = cv2.cvtColor(image_matrix, cv2.COLOR_GRAY2BGR)
    output = remove(image,session=session)
    alpha_channel = output[:, :, 3]
    return jsonify({'result': alpha_channel.tolist()})

if __name__ == '__main__':
    app.run()