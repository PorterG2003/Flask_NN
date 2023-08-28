#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 14 15:24:27 2023

@author: porter
"""

from mlxtend.data import loadlocal_mnist
import platform
import numpy as np
from nn import NeuralNet as NN
import time
from random import randint

class TrainingTests:
    
    def __init__(self):
        print("Loading Data")
        #Load Data
        if not platform.system() == 'Windows':
            self.train_X, self.train_y = loadlocal_mnist(
                    images_path='Database/train-images-idx3-ubyte', 
                    labels_path='Database/train-labels-idx1-ubyte')
            self.test_X, self.test_y = loadlocal_mnist(
                    images_path='Database/t10k-images-idx3-ubyte', 
                    labels_path='Database/t10k-labels-idx1-ubyte')
        
        else:
            self.train_X, self.train_y = loadlocal_mnist(
                    images_path='tDatabase/rain-images.idx3-ubyte', 
                    labels_path='Database/train-labels.idx1-ubyte')
            self.test_X, self.test_y = loadlocal_mnist(
                    images_path='Database/t10k-images.idx3-ubyte', 
                    labels_path='Database/t10k-labels.idx1-ubyte')
        
        print("Formatting Data")
        # Format labels from decimal to one hot encoded
        self.train_y_formatted = []
        self.test_y_formatted = []
        for l in self.train_y:
            ni = [0,0,0,0,0,0,0,0,0,0]
            ni[l] = 1
            self.train_y_formatted.append(ni)
        for l in self.test_y:
            ni = [0,0,0,0,0,0,0,0,0,0]
            ni[l] = 1
            self.test_y_formatted.append(ni)
        
    
    def test_back_propagation(self, out=False):
        print("\nTesting Back Propagation...")
        nn = NN(self.train_X, self.train_y_formatted, self.test_X, self.test_y_formatted)
        nn.save()
        
        for i in range(5):
            start_index = randint(0, (len(nn.X)-1000)//1000)*1000
            end_index = start_index + 1000
            
            loss1 = np.average(nn.crossEntropyLoss())
            if np.isnan(loss1):
                print(nn.crossEntropyLoss())
            
            print(f"\n\nperforming back propagation from index {start_index} to {end_index}")
            nn.backPropagation(start_index=start_index, end_index=end_index)
            nn.forward(nn.X)
            
            loss2 = np.average(nn.crossEntropyLoss())
            if np.isnan(loss2):
                print(nn.crossEntropyLoss())
            
            if out:
                print("average loss:", loss1," => ", loss2)
                
            if loss1 < loss2:
                print("Failed")
                #return False
        
        print("Passed")
        return True
            
    
if __name__ == '__main__':
    test = TrainingTests()
    passes = []
    passes.append(test.test_back_propagation(out=True))
    print("\nPassed "+str(passes.count(True))+"/"+str(len(passes))+" tests")
    
            