#!/usr/bin/env python3

"""
This file defines some neuron types for use in a neural network.
Created by Joseph Camacho
"""

import random
from functions import sigmoid

class Neuron:
    """
    Represents an ordinary neuron in a neural network
    """
    def __init__(self, inputs, outputs, input_weights, bias):
        self.inputs = inputs
        self.outputs = outputs
        self.input_weights = input_weights
        self.bias = bias

        self.reset()

    def reset(self):
        """
        Prepare a neuron for both firing and back propagation
        """
        self.reset_fire()
        self.reset_back_propagate()
    
    def reset_fire(self):
        """
        Prepare a neuron for firing again
        """
        self.fired = False
        self.fire_value = 0

    def reset_back_propagate(self):
        """
        Prepare a neuron for back propagating again
        """
        self.back_propagated = False
        self.back_propagate_value = 0

    def get_weight(self, neuron):
        """
        Return the weight connecting an earlier neuron to this neuron
        """
        return self.input_weights[self.inputs.index(neuron)]

    def fire(self):
        """
        Have the neuron fire
        """
        if self.fired:
            return self.fire_value
        
        self.fired = True
        weighted_sum = sum(self.inputs[i].fire() * self.input_weights[i] for i in range(len(self.input_weights)))
        self.fire_value = sigmoid(weighted_sum + self.bias)
        return self.fire_value

    def back_propogate(self):
        """
        Return the product of the partial derivatives up
        to this neuron. (This is the value the weights coming in
        should be multiplied by when using gradient descent.)
        """
        if self.back_propagated:
            return self.back_propagate_value

        self.back_propagated = True
        weighted_sum = sum(output.back_propogate() * output.get_weight(self) for output in self.outputs)
        self.back_propogate_value = weighted_sum * self.fire_value * (1 - self.fire_value)
        return self.back_propogate_value

    def update_weights(self, step_size=0.1):
        """
        Update this neuron's weights and bias using gradient descent
        with the partial derivatives calculated during
        back propagation.
        """
        for i in range(len(self.input_weights)):
            self.input_weights[i] -= self.neurons[i].fire() * self.back_propogate_value * step_size

        self.bias -= self.back_propogate_value * step_size
        

class Input:
    """
    This is similar to a neuron, but lacks many of its functions.
    Its purpose is to provide an input to a neural network.
    """
    def __init__(self, value=None):
        self.value = value

    def set_value(self, value):
        self.value = value

    def reset(self):
        pass

    def fire(self):
        return self.value

class OutputNeuron(Neuron):
    """
    Neuron representing the end -- almost identical to a normal neuron,
    but it has no outputs
    """
    def __init__(self, inputs, input_weights, bias):
        self.inputs = inputs
        self.input_weights = input_weights
        self.bias = bias

        self.reset()

    def back_propogate(self, expected):
        if self.back_propagated:
            return self.back_propagate_value

        self.back_propagated = True
        self.back_propogate_value = (expected - self.fire_value) * self.fire_value * (1 - self.fire_value)
        return self.back_propogate_value

    
class RandomNeuron(Neuron):
    """
    Neuron with random initial state
    """
    def __init__(self, inputs, outputs):
        input_weights = [2 * random.random() - 1 for i in range(len(inputs))]
        bias = 2 * random.random() - 1
        super().__init__(inputs, outputs, input_weights, bias)

class RandomOutputNeuron(OutputNeuron):
    """
    Output neuron with random initial state
    """
    def __init__(self, inputs):
        input_weights = [2 * random.random() - 1 for i in range(len(inputs))]
        bias = 2 * random.random() - 1
        super().__init__(inputs, input_weights, bias)
