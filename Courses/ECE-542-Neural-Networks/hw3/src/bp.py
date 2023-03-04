#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: bp.py

import numpy as np
from src.activation import sigmoid, sigmoid_prime

def backprop(x, y, biases, weightsT, cost, num_layers):
    """ function of backpropagation
        Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient of all biases and weights.

        Args:
            x, y: input image x and label y
            biases, weights (list): list of biases and transposed weights of entire network
            cost (CrossEntropyCost): object of cost computation
            num_layers (int): number of layers of the network

        Returns:
            (nabla_b, nabla_wT): tuple containing the gradient for all the biases
                and weightsT. nabla_b and nabla_wT should be the same shape as 
                input biases and weightsT
    """
    # initial zero list for store gradient of biases and weights
    nabla_b = [np.zeros(b.shape) for b in biases]
    nabla_wT = [np.zeros(wT.shape) for wT in weightsT]

    activations = []
    activations.append(x)
    for k in range(0,num_layers-1):
        activations.append(sigmoid(np.dot(weightsT[k],activations[k])+biases[k]))

    delta = (cost).df_wrt_a(activations[-1], y)

    for i in range(num_layers-2,-1,-1):
     
         if i == num_layers-2:
            nabla_b[i] = delta
            nabla_wT[i] = np.dot(delta,np.transpose(activations[-2]))
         else:
            delta = np.dot(np.transpose(weightsT[i+1]),delta)*sigmoid_prime(np.dot(weightsT[i],activations[i])+biases[i])
            nabla_b[i] = delta
            nabla_wT[i] = np.dot(delta,np.transpose(activations[i]))

    return (nabla_b, nabla_wT)

