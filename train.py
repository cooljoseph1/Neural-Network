#!/usr/bin/env python3

"""
This file is used to train a neural network to give outputs given inputs.
"""

def distance_squared(vector1, vector2):
    return sum((x1 - x2)**2 for x1, x2 in zip(vector1, vector2))

def train(neural_network, training_data, step_size=10, desired_error=0.1, max_turns=10000, verbose=False):
    """
    Train a neural network given training data.
    step_size is the maximum step size allowed, but this may decrease as
    the optimum is approached.  The desired error is the maximum error
    you want to allow in your neural network.  Max turns is the maximum
    number of turns allowed in training, and dynamic_step_size can be
    set to False to get rid of changing the step size.  Verbosity prints
    additional information while training occurs.
    """

    old_error = len(training_data[0][1])*len(training_data)
    error = old_error
    for turn in range(max_turns):
        error = 0
        for inputs, desired_outputs in training_data:
            # Test the neural network
            neural_network.reset()
            outputs = neural_network.compute_outputs(inputs)
            error += distance_squared(outputs, desired_outputs)
            
            # Back propagate to compute differentials
            neural_network.back_propagate(desired_outputs, step_size)

        # Break if we achieve the desired error
        if error <= desired_error * len(training_data):
            break

        neural_network.set_new_weights()
        old_error = error
        
        if verbose:
            print(error, step_size)

    print("Completed training in {} turns with an error of {}".format(turn, error))
            
    return error
