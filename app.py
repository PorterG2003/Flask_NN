# 07-16-2023 Porter Gardiner

from flask import Flask, render_template, jsonify, request
from nn_initializer import init_NN
import numpy as np
from helpers import scale_weights
from trainer import Trainer
import threading
import logging
import sys

app = Flask(__name__)

logger = logging.getLogger('Computation')
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler('train.log', mode='w')
stderr_handler = logging.StreamHandler(sys.stderr)
formatter = logging.Formatter('%(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
stderr_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(stderr_handler)

nn = init_NN(logger=logger)
background_thread = threading.Thread()
trainer = Trainer(nn, logger=logger)

@app.route('/api/data', methods=['GET', 'POST'])
def get_data():
    weights = [
        nn.W1.tolist(),
        nn.W2.tolist(),
        nn.W3.tolist(),
    ]

    scaled_weights = scale_weights(weights)

    data = {
        'layer_sizes': [
            nn.inputLayerSize,
            nn.hiddenLayerSize1,
            nn.hiddenLayerSize2,
            nn.outputLayerSize,
        ],
        'weights': scaled_weights
    }
    return jsonify(data)

@app.route('/api/images')
def get_images():
    weights = [
        nn.W1.tolist(),
        nn.W2.tolist(),
        nn.W3.tolist(),
    ]

    scaled_weights = scale_weights(weights)

    images = [
        scaled_weights,
    ]
    return jsonify(images)

@app.route('/nn/start', methods=['POST'])
def start_training():
    if not trainer.running:
        # Get the learning_rate from the query parameters or provide a default value
        data = request.get_json()
        learning_rate = float(data.get('learning_rate', 0.01))
        print(f"Recieved learning rate: {learning_rate}")
        batch_size = int(data.get('batch_size', 256))
        print(f"Recieved batch size: {batch_size}")

        # Start the background thread for training
        background_thread = threading.Thread(target=trainer.SGD, args=(batch_size, learning_rate))
        background_thread.start()

        response = {'response': 'Started training'}
    else:
        response = {'response': 'Failed to start training. Already Training.'}
    return jsonify(response)

@app.route('/nn/stop')
def stop_training():
    if trainer.running:
        trainer.stop_training()
        response = {'response': 'Stopped training'}
    else:
        response = {'response': 'Failed to stop training. No training in progress to stop.'}
    return jsonify(response)

@app.route('/nn/new')
def new_nn():
    global nn, background_thread, trainer
    stop_training()

    nn = init_NN()
    trainer = Trainer(nn)

    response = {'response': 'Created new network'}
    return response

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run()
