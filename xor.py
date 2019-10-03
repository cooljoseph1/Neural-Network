#!/usr/bin/env python3

"""
This is an example of using a neural network to learn the xor function.
The sample neural network has 3 layers:  an input layer of 2 neurons, a
hidden layer of 2 neurons, and an output layer of 1 neuron.
Created by Joseph Camacho
"""

from train import train
from network import Network

training_data = [
    ([0, 0], [0]),
    ([0, 1], [1]),
    ([1, 0], [1]),
    ([1, 1], [0]),
]

xor_network = Network([2, 2, 1])
print(train(xor_network, training_data))
for sample in training_data:
    print(xor_network.quick_calculate(sample[0]))
