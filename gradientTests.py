#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 24 16:55:22 2022

@author: porter
"""

from mlxtend.data import loadlocal_mnist
import platform
import numpy as np
from nn import NeuralNet as NN
import time

class GradientTests:
    
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
        
    
    def test_dCdW3(self, instance, out=False):
        print("\nTesting dCdW3...")
        nn = NN(self.train_X, self.train_y_formatted, self.test_X, self.test_y_formatted)
        learning_rate = 0.0001
        
        for i in range(5):
            loss1 = nn.crossEntropyLossSingle(instance)
            nn.forward(nn.X)
            dCdW3 = nn.calculate_dCdW3(instance)
            formatted = np.empty(nn.W3.shape)
            for k in range(len(dCdW3[0])):
                i = k % nn.W3.shape[0]
                j = k // nn.W3.shape[0]
                formatted[i][j] = dCdW3[0][k]
            nn.W3 = nn.W3 - learning_rate * formatted
            loss2 = nn.crossEntropyLossSingle(instance)
            
            if out:
                print("loss:", loss1," => ", loss2)
                
            if loss1 < loss2:
                print("Failed")
                return False
        
        print("Passed")
        return True
    
    def test_dCdW3_quick(self, instance, out=False):
        print("\nTesting dCdW3 quick...")
        nn = NN(self.train_X, self.train_y_formatted, self.test_X, self.test_y_formatted)
        learning_rate = 0.0001
        
        for i in range(5):
            loss1 = nn.crossEntropyLossSingle(instance)
            nn.forward(nn.X)
            nn.W3 = nn.W3 - learning_rate * nn.calculate_dCdW3_quick(instance)
            loss2 = nn.crossEntropyLossSingle(instance)
            
            if out:
                print("loss:", loss1," => ", loss2)
                
            if loss1 < loss2:
                print("Failed")
                return False
        
        print("Passed")
        return True
    
    def test_dCdB3(self, instance, out=False):
        print("\nTesting dCdB3...")
        nn = NN(self.train_X, self.train_y_formatted, self.test_X, self.test_y_formatted)
        learning_rate = 1
        
        for i in range(5):
            loss1 = nn.crossEntropyLossSingle(instance)
            nn.forward(nn.X)
            nn.B3 = nn.B3 - learning_rate * nn.calculate_dCdB3(instance)
            loss2 = nn.crossEntropyLossSingle(instance)
            
            if out:
                print("loss:", loss1," => ", loss2)
                
            if loss1 < loss2:
                print("Failed")
                return False
        
        print("Passed")
        return True
    
    def test_dCdW2(self, instance, out=False):
        print("\nTesting dCdW2...")
        nn = NN(self.train_X, self.train_y_formatted, self.test_X, self.test_y_formatted)
        learning_rate = 0.0001
        
        for i in range(5):
            loss1 = nn.crossEntropyLossSingle(instance)
            nn.forward(nn.X)
            dCdW2 = nn.calculate_dCdW2(instance)
            formatted = np.empty(nn.W2.shape)
            for k in range(len(dCdW2[0])):
                i = k % nn.W2.shape[0]
                j = k // nn.W2.shape[0]
                formatted[i][j] = dCdW2[0][k]
            nn.W2 = nn.W2 - learning_rate * formatted
            loss2 = nn.crossEntropyLossSingle(instance)
            
            if out:
                print("loss:", loss1," => ", loss2)
                
            if loss1 < loss2:
                print("Failed")
                return False
        
        print("Passed")
        return True
    
    def test_dCdB2(self, instance, out=False):
        print("\nTesting dCdB2...")
        nn = NN(self.train_X, self.train_y_formatted, self.test_X, self.test_y_formatted)
        learning_rate = 1
        
        for i in range(5):
            loss1 = nn.crossEntropyLossSingle(instance)
            nn.forward(nn.X)
            nn.B2 = nn.B2 - learning_rate * nn.calculate_dCdB2(instance)
            loss2 = nn.crossEntropyLossSingle(instance)
            
            if out:
                print("loss:", loss1," => ", loss2)
                
            if loss1 < loss2:
                print("Failed")
                return False
        
        print("Passed")
        return True
    
    def test_dCdW1(self, instance, out=False):
        print("\nTesting dCdW1...")
        nn = NN(self.train_X, self.train_y_formatted, self.test_X, self.test_y_formatted)
        learning_rate = 0.0001
        
        for i in range(5):
            loss1 = nn.crossEntropyLossSingle(instance)
            nn.forward(nn.X)
            dCdW1 = nn.calculate_dCdW1(instance)
            formatted = np.empty(nn.W1.shape)
            for k in range(len(dCdW1[0])):
                i = k % nn.W1.shape[0]
                j = k // nn.W1.shape[0]
                formatted[i][j] = dCdW1[0][k]
            nn.W1 = nn.W1 - learning_rate * formatted
            loss2 = nn.crossEntropyLossSingle(instance)
            
            if out:
                print("loss:", loss1," => ", loss2)
                
            if loss1 < loss2:
                print("Failed")
                return False
        
        print("Passed")
        return True
    
    def test_dCdB1(self, instance, out=False):
        print("\nTesting dCdB1...")
        nn = NN(self.train_X, self.train_y_formatted, self.test_X, self.test_y_formatted)
        learning_rate = 1
        
        for i in range(5):
            loss1 = nn.crossEntropyLossSingle(instance)
            nn.forward(nn.X)
            nn.B1 = nn.B1 - learning_rate * nn.calculate_dCdB1(instance)
            loss2 = nn.crossEntropyLossSingle(instance)
            
            if out:
                print("loss:", loss1," => ", loss2)
                
            if loss1 < loss2:
                print("Failed")
                return False
        
        print("Passed")
        return True
            
    
if __name__ == '__main__':
    test = GradientTests()
    passes = []
    passes.append(test.test_dCdW3_quick(0, out=True))
    passes.append(test.test_dCdW3(0, out=True))
    passes.append(test.test_dCdB3(0, out=True))
    passes.append(test.test_dCdW2(0, out=True))
    passes.append(test.test_dCdB2(0, out=True))
    passes.append(test.test_dCdW1(0, out=True))
    passes.append(test.test_dCdB1(0, out=True))
    print("\nPassed "+str(passes.count(True))+"/"+str(len(passes))+" tests")
    
            