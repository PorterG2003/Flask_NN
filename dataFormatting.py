#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 22:16:39 2019

@author: porterg2003
"""


from mnist import MNIST
mndata = MNIST('./Database')
train_images, train_labels =  mndata.load_training()
test_images, test_labels = mndata.load_testing()
train_labels_formatted = []
test_labels_formatted = []
ni = [0,0,0,0,0,0,0,0,0,0]
for l in train_labels:
    ni[l] = 1
    train_labels_formatted.append(ni)
    ni = [0,0,0,0,0,0,0,0,0,0]
for l in test_labels:
    ni[l] = 1
    test_labels_formatted.append(ni)
    ni = [0,0,0,0,0,0,0,0,0,0]
    
