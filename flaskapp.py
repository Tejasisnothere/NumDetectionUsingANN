from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
from scipy.ndimage import gaussian_filter, center_of_mass, shift

app = Flask(__name__)

with open('./Numdetection/model.pkl', 'rb') as f:
    gw, gb = pickle.load(f)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    matrix = np.array(data['matrix'], dtype=np.float32).reshape(28, 28) / 255.0

    matrix = gaussian_filter(matrix, sigma=0.8)
    matrix = center_image(matrix)
    matrix[matrix < 0.05] = 0.0

    inp = matrix.flatten()

    probabilities = predictProbabilities(inp, gw, gb)
    prediction = int(np.argmax(probabilities))

    return jsonify({
        'prediction': prediction,
        'probabilities': [round(p, 4) for p in probabilities.tolist()]
    })


def center_image(img):
    cy, cx = center_of_mass(img)
    shift_y = np.round(img.shape[0]/2 - cy)
    shift_x = np.round(img.shape[1]/2 - cx)
    return shift(img, shift=(shift_y, shift_x), mode='constant')

def predictProbabilities(inp, ws, bs):
    ro1 = inp @ ws[0].T + bs[0]
    ao1 = leakyRELU(ro1)
    ro2 = ao1 @ ws[1].T + bs[1]

    # Softmax for probabilities
    exps = np.exp(ro2 - np.max(ro2))
    probs = exps / np.sum(exps)
    return probs


def leakyRELU(arr):
    return np.where(arr > 0, arr, 0.01 * arr)


def predictOutput(inp, ws, bs):
    ro1 = inp @ ws[0].T + bs[0]
    ao1 = leakyRELU(ro1)
    ro2 = ao1 @ ws[1].T + bs[1]
    return int(np.argmax(ro2))


if __name__ == "__main__":
    app.run(debug=True)
