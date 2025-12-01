#! /usr/bin/env python3
from nn.model import NeuralNetwork
from nn.layer import Dense
from nn.activation import ReLU, Sigmoid
from nn.loss import BinaryCrossEntropy
from nn.optimizer import Adam
from compress_nn import compress_model_object


model = NeuralNetwork(
    layers=(
        (Dense(769), ReLU()),
        (Dense(128), ReLU()),
        (Dense(64), ReLU()),
        (Dense(5), Sigmoid()),
    ),
    loss=BinaryCrossEntropy(),
    optimizer=Adam(learning_rate=0.001),
    regularization_factor=0.001,
)
compress_model_object(model)
model.save("test.nn")

