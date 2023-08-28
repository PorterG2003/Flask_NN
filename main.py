#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 10:00:59 2022

@author: porterg2003
"""

from mlxtend.data import loadlocal_mnist
import platform
import numpy as np
from nn import NeuralNet as NN

print("Loading Data")
#Load Data
if not platform.system() == 'Windows':
    train_X, train_y = loadlocal_mnist(
            images_path='Database/train-images-idx3-ubyte', 
            labels_path='Database/train-labels-idx1-ubyte')
    test_X, test_y = loadlocal_mnist(
            images_path='Database/t10k-images-idx3-ubyte', 
            labels_path='Database/t10k-labels-idx1-ubyte')

else:
    train_X, train_y = loadlocal_mnist(
            images_path='tDatabase/rain-images.idx3-ubyte', 
            labels_path='Database/train-labels.idx1-ubyte')
    test_X, test_y = loadlocal_mnist(
            images_path='Database/t10k-images.idx3-ubyte', 
            labels_path='Database/t10k-labels.idx1-ubyte')

print("Formatting Data")
# Format labels from decimal to one hot encoded
train_y_formatted = []
test_y_formatted = []
for l in train_y:
    ni = [0,0,0,0,0,0,0,0,0,0]
    ni[l] = 1
    train_y_formatted.append(ni)
for l in test_y:
    ni = [0,0,0,0,0,0,0,0,0,0]
    ni[l] = 1
    test_y_formatted.append(ni)

print("Initializing Neural Network")
nn = NN(train_X, train_y_formatted, test_X, test_y_formatted)

nn.batchGradientDescent(nn.X, nn.y, learning_rate=.01)
nn.classify(test_X[1000:4000:100], test_y_formatted[1000:4000:100])
