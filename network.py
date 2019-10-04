#!/usr/bin/env python3

"""
This file defines a simple neural network.
Created by Joseph Camacho
"""

from neuron import Input, RandomNeuron, RandomOutputNeuron

class Network:
    """
    A neural network
    """
    def __init__(self, layer_sizes):
        self.neurons = []
        # Create neurons
        for i, layer_size in enumerate(layer_sizes):
            if i == 0:
                self.neurons.append(
                    [Input() for _ in range(layer_size)]
                )
            elif i < len(layer_sizes) - 1:
                self.neurons.append(
                    [RandomNeuron(self.neurons[-1], []) for _ in range(layer_size)]
                )
            else:
                self.neurons.append(
                    [RandomOutputNeuron(self.neurons[-1]) for _ in range(layer_size)]
                )

        # Point neurons to correct output neurons
        for i, layer in enumerate(self.neurons):
            if i == 0 or i == len(self.neurons) - 1:
                continue
            
            for neuron in layer:
                neuron.outputs = self.neurons[i + 1]

    def reset(self):
        """
        Reset the neurons in the neural network and prepare for new inputs
        """
        self.outputs = None
        
        for layer in self.neurons:
            for neuron in layer:
                neuron.reset()
    

    def compute_outputs(self, inputs):
        """
        Calculate the outputs for given inputs
        """        
        for input, input_neuron in zip(inputs, self.neurons[0]):
            input_neuron.set_value(input)

        outputs = [output_neuron.fire() for output_neuron in self.neurons[-1]]        
        return outputs

    def back_propagate(self, desired_outputs, step_size=0.1):
        """
        Back propagate the gradients
        """
        for output_neuron, desired_output in zip(self.neurons[-1], desired_outputs):
            output_neuron.back_propagate(desired_output, step_size=step_size)

        for layer in self.neurons[-2:0:-1]:
            for neuron in layer:
                neuron.back_propagate(step_size=step_size)

    def set_new_weights(self):
        """
        Update the weights
        """
        for layer in self.neurons[1:]:
            for neuron in layer:
                neuron.set_new_weights()

    def quick_calculate(self, inputs):
        """
        Reset then compute outputs for the inputs
        """
        self.reset()
        return self.compute_outputs(inputs)
