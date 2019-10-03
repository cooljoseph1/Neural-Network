#!/usr/bin/env python3

"""
This file is used to train a neural network to give outputs given inputs.
"""

def distance_squared(vector1, vector2):
    return sum((x1 - x2)**2 for x1, x2 in zip(vector1, vector2))

def train(neural_network, training_data, desired_error=0.1, max_turns=10000):
    # dynamic step size
    step_size = 0.1
    old_distance = len(training_data[0][1])
    distance = old_distance
    for turn in range(max_turns):
        for inputs, desired_outputs in training_data:
            neural_network.reset()
            outputs = neural_network.compute_outputs(inputs)

            # Break if we achieve the desired error
            distance = distance_squared(outputs, desired_outputs)
            if distance <= desired_error * len(desired_outputs):
                break

            # Adjust step size so that we take bigger steps when possible,
            #  but try not to overshoot
            if distance < old_distance:
                step_size *= 1.05
            else:
                step_size *= 0.5
            
            old_distance = distance
            
            # Back propogate and update weights
            neural_network.back_propogate(desired_outputs)
            neural_network.update_weights(step_size)
            
    return distance
