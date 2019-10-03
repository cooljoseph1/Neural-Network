#!/usr/bin/env python3

"""
This file defines some basic functions to be used elsewhere.
Created by Joseph Camacho
"""

import math

def sigmoid(x):
    """
    Basic sigmoid function
    """
    return 1 / (1 + math.exp(-x))
