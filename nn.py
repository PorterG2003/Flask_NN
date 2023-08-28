#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 09:57:15 2022

@author: porterg2003
"""

#TODO:
'''
    have tests save nn and be able to load old ones.
'''

import numpy as np
import time
import datetime
import os
import pickle
import logging
np.seterr(all='raise')
np.seterr(under='warn')

# ***** Neural Network *****
class NeuralNet:
    def __init__(self, X, y, X_test, y_test, logger=None):
        if not logger:
            # Set up the custom logger for this class
            self.logger = logging.getLogger('Trainer')
            self.logger.setLevel(logging.INFO)

            # Create a file handler for the log file
            file_handler = logging.FileHandler('trainer.log')

            # Create a formatter to specify the log message format
            formatter = logging.Formatter('%(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)

            # Add the file handler to the logger
            self.logger.addHandler(file_handler)
        else:
            self.logger = logger

        self.inputLayerSize = 784
        self.hiddenLayerSize1 = 28
        self.hiddenLayerSize2 = 28
        self.outputLayerSize = 10

        #Set inputs
        self.X = np.array(X, copy=True)/255
        self.y = np.array(y, copy=True)

        self.X_test = np.array(X_test, copy=True)
        self.y_test = np.array(y_test, copy=True)

        # Initialize weights using Xavier initialization
        sigma1 = np.sqrt(2 / self.inputLayerSize)
        sigma2 = np.sqrt(2 / self.hiddenLayerSize1)
        sigma3 = np.sqrt(2 / self.hiddenLayerSize2)

        self.W1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize1) * sigma1
        self.W2 = np.random.randn(self.hiddenLayerSize1, self.hiddenLayerSize2) * sigma2
        self.W3 = np.random.randn(self.hiddenLayerSize2, self.outputLayerSize) * sigma3

        self.B1 = np.zeros(self.hiddenLayerSize1)
        self.B2 = np.zeros(self.hiddenLayerSize2)
        self.B3 = np.zeros(self.outputLayerSize)

        self.initials = {"W1": self.W1, "W2": self.W2, "W3": self.W3, "B1": self.W1, "B2": self.W2, "B3": self.W3}

        self.yhat = self.forward(self.X)


        #-------HELPER DATA MEMBERS--------


    def save(self, notes=None):
        # Create directory name
        now = datetime.datetime.now()
        dir_name = "saved_nns/{}".format(now.strftime("%Y-%m-%d-%H-%M"))

        # Create directory if it does not exist
        os.makedirs(dir_name, exist_ok=True)

        # Save NeuralNet object as .pkl file
        file_name = os.path.join(dir_name, "model.pkl")
        with open(file_name, "wb") as f:
            pickle.dump(self, f)
            
        if notes:
            with open("Output.txt", "w") as text_file:
                text_file.write(notes)
    
    def load_initials(self):
        self.W1 = self.initials["W1"]
        self.B1 = self.initials["B1"]
        self.W2 = self.initials["W2"]
        self.B2 = self.initials["B2"]
        self.W3 = self.initials["W3"]
        self.B3 = self.initials["B3"]
        

    def forward(self, X):
        if np.isnan(X).any() or np.isnan(self.W1).any() or np.isnan(self.W2).any() or np.isnan(self.W3).any():
            self.logger.info("Input or weights contain NaNs.")
    
        self.Z2 = np.dot(X, self.W1)
        self.Z2 += self.B1
        if np.isnan(self.Z2).any():
            self.logger.info("NaN detected in Z2")
            
        self.A2 = self.reLu(self.Z2)
        if np.isnan(self.A2).any():
            self.logger.info("NaN detected in A2")
            
        self.Z3 = np.dot(self.A2, self.W2)
        self.Z3 += self.B2
        if np.isnan(self.Z3).any():
            self.logger.info("NaN detected in Z3")
            
        self.A3 = self.reLu(self.Z3)
        if np.isnan(self.A3).any():
            self.logger.info("NaN detected in A3")
            
        self.Z4 = np.dot(self.A3, self.W3)
        self.Z4 += self.B3
        if np.isnan(self.Z4).any():
            self.logger.info("NaN detected in Z4")
            
        self.yhat = self.softMax(self.Z4)
        if np.isnan(self.yhat).any():
            self.logger.info("NaN detected in yhat")
            
        return np.array(self.yhat, copy=True)
    
    
    def forward_test(self, X):
        Z2 = np.dot(X, self.W1)
        Z2 += self.B1
        
        A2 = self.reLu(Z2)
        
        Z3 = np.dot(A2, self.W2)
        Z3 += self.B2
        
        A3 = self.reLu(Z3)
        
        Z4 = np.dot(A3, self.W3)
        Z4 += self.B3
        
        yhat = self.softMax(Z4)
        return yhat
    

    def reLu(self, Z):
        #make copy of parameters
        z = np.array(Z, copy=True)
        
        #compute reLu
        for i in range(len(z)):
            for x in range(len(z[i])):
                z[i][x] = z[i][x] * (z[i][x] > 0)
        return z
    
    
    #computes softmax for every instance in Z
    #returns array with same shape as Z
    def softMax(self, Z):
        #make copy of parameters
        z = np.array(Z, copy=True)
        
        #for 2 dimensional arrays
        if isinstance(z[0], (np.ndarray, list)):
            for i in range(len(z)):
                #prevent overflow
                m = max(z[i])
                #print(z[i])
                for j in range(len(z[i])):
                    z[i][j] = z[i][j] - m
                    #e to the power of z[i][j]
                    try:
                        #print(z[i][j])
                        z[i][j] = np.exp(z[i][j])
                    #if result too close to zero
                    except:
                        self.logger.info(f"np.exp(z[i][j]) could not be computed \nz[i][j]: {z[i][j]}")
                        #set to just barely above zero to prevent dividing by zero in crossEntropyLoss()
                        z[i][j] = 1e-15
                    
                #divide by sum
                #print(z[i])
                #input()
                z[i] = z[i]/np.sum(z[i])
        
        #for 1 dimensional arrays
        else:
            m = max(z)
            #prevent overflow
            for i in range(len(z)):
                z[i] = z[i] - m
                #e to the z[i]
                try:
                    z[i] = np.exp(z[i])
                #if result too close to zero
                except:
                    self.logger.info(f"np.exp(z[i]) could not be computed \nz[i]: {z[i]}")
                    #set to just barely above zero to prevent dividing by zero in crossEntropyLoss()
                    z[i] = 1e-15
            
            #softmax
            #print(z)
            #input()
            z = z/np.sum(z)
            
        return z
    
    
    def crossEntropyLossSingle(self, instance):
        
        count = 0
        for j in range(len(self.y[instance])):
            if self.y[instance][j] == 1:
                #added 1e-15 to prevent underflow
                d = -np.log10(self.yhat[instance][j] + 1e-15)
                count += 1
                
        #raise ValueError if y[i] has more or less than 1 correct class
        if count < 1:
            raise ValueError("No Correct Value in y["+str(instance)+"]")
        if count > 1:
            raise ValueError("Multiple Correct Values in y["+str(instance)+"]")
            
        return d
    

    #computes crossEntropyLoss for every instance
    def crossEntropyLoss(self):
        
        #make empty array with shape
        d = np.empty((len(self.X),1))
        
        #compute
        for i in range(len(d)):
            count = 0
            for j in range(len(self.y[i])):
                if self.y[i][j] == 1:
                    try:
                        #added 1e-15 to prevent underflow
                        d[i] = -np.log10(self.yhat[i][j] + 1e-15)
                    except FloatingPointError:
                        raise FloatingPointError(f"-np.log10(self.yhat[{i}][{j}]) could not be computed\nself.yhat[{i}][{j}]: {self.yhat[i][j]}")
                    if np.isnan(d[i]):
                        self.logger.info(self.yhat[i])
                        input()
                    count += 1
                    
            #raise ValueError if y[i] has more or less than 1 correct class
            if count < 1:
                raise ValueError("No Correct Value in y["+str(i)+"]")
            if count > 1:
                raise ValueError("Multiple Correct Values in y["+str(i)+"]")
        
        return d
    
    
    def calculate_dZ2dW1(self, instance):
        Z2_length = len(self.Z2[instance])
        X_length = len(self.X[instance])
        
        #make zero array with shape
        jac = np.zeros((Z2_length,X_length*Z2_length))
        
        #compute gradients
        for i in range(len(jac)):
            for j in range(X_length):
                jac[i][j+(i*X_length)] = self.X[instance][j]
                
        return jac
    
    
    def calculate_dA2dZ2(self, instance):
        jac = np.zeros((len(self.Z2[0]),len(self.Z2[0])))
        
        for i in range(len(self.Z2[0])):
            if self.Z2[instance][i] >= 0:
                jac[i][i] = 1
        
        return jac
    
    
    def calculate_dZ3dA2(self, instance):
        return self.W2.T
    
    
    def calculate_dZ3dW2(self, instance):
        Z3_length = len(self.Z3[instance])
        A2_length = len(self.A2[instance])
        
        #make zero array with shape
        jac = np.zeros((Z3_length,A2_length*Z3_length))
        
        #compute gradients
        for i in range(len(jac)):
            for j in range(A2_length):
                jac[i][j+(i*A2_length)] = self.A2[instance][j]
                
        return jac
    
    
    def calculate_dA3dZ3(self, instance):
        jac = np.zeros((len(self.Z3[0]),len(self.Z3[0])))
        
        for i in range(len(self.Z3[0])):
            if self.Z3[instance][i] >= 0:
                jac[i][i] = 1
        
        return jac
    
    
    def calculate_dZ4dA3(self, instance):
        return self.W3.T
    
    
    #Jacobian of Z4 with respect to W3
    #Takes in one instance of Z4 and A3
    def calculate_dZ4dW3(self, instance):
        #make copy of parameters
        Z4_length = len(self.Z4[instance])
        A3_length = len(self.A3[instance])
        
        #make zero array with shape
        jac = np.zeros((Z4_length,A3_length*Z4_length))
        
        #compute gradients
        for i in range(len(jac)):
            for j in range(A3_length):
                jac[i][j+(i*A3_length)] = self.A3[instance][j]
                
        return jac
    
    
    #Jacobian of yhat with respect to Z4
    #Takes in one instance of yhat and Z4
    def calculate_dyhatdZ4(self, instance):
        #compute softmax of the instance of Z4
        softmax_Z4 = self.yhat[instance]
        
        #set variables to length of the instance of Z4 and yhat
        Z4_length = len(self.Z4[instance])
        
        #make zero array with correct shape
        jac = np.zeros((Z4_length, Z4_length))
        
        #compute jacobian
        for i in range(Z4_length):
            for j in range(Z4_length):
                if i == j:
                    jac[i][j] = softmax_Z4[i]*(1-softmax_Z4[j])
                else:
                    jac[i][j] = -softmax_Z4[j]*softmax_Z4[i]
                    
        return jac
    
    
    #Jacobian of crossEntropyLoss with respect to yhat
    def calculate_dCdyhat(self, instance):
        #make zero array with shape (1, length of instance of yhat)
        jac = np.zeros((1, len(self.yhat[0])))
        
        #compute jacobian
        count = 0
        for j in range(len(self.y[instance])):
            if self.y[instance][j] == 1:
                if self.yhat[instance][j] != 0:
                    jac[0][j] = -1/self.yhat[instance][j]
                else:
                    self.logger.info(f"jac[0][{j}] = -1/yhat[{instance}][{j}] could not be preformed as yhat[{instance}][{j}] == 0")
                    raise ValueError
                count += 1
         
        #throw ValueError when y has more or less than one positive class
        if count < 1:
            raise ValueError("No Correct Value in y["+str(instance)+"]")
        if count > 1:
            raise ValueError("Multiple Correct Values in y["+str(instance)+"]")
        
        return jac
    
        
    #Jacobian of crossEntropyLoss with respect to W3 for one instance
    def calculate_dCdW3(self, instance):

        #compute jacobians
        d1 = self.calculate_dCdyhat(instance)
        d2 = self.calculate_dyhatdZ4(instance)
        d3 = self.calculate_dZ4dW3(instance)
        
        #dot jacobians together
        dyhatdW3 = np.dot(d2, d3)
        dCdW3 = np.dot(d1, dyhatdW3)
        
        return dCdW3
    

    #averages gradient over every instance then formats the average to the shape of given array
    #returns formatted average with shape of given array
    def calculate_gradient_average(self, array, calc_function, indices=None):
        tt1 = time.time()

        if indices is None:
            indices = [i for i in range(len(self.X))]

        #get shape of an instance and set total to zero array with shape (1, i, j)
        self.logger.info(f"Array shape: {array.shape}")
        try:
            total = np.empty((1, array.shape[0], array.shape[1]))
        except IndexError:
            total = np.empty(array.shape[0])

        #calculate total
        t1 = time.time()
        for i in indices:
            #np.append(total, self.calculate_dCdW3_quick(i))
            np.append(total, calc_function(i))

        #calculate average
        average = np.average(total, axis=0)
        t2 = time.time()
        #print("time to calculate batch:",t2-t1)

        #format the average
        try:
            average_formatted = np.empty(array.shape)
            for k in range(len(average[0])):
                i = k % array.shape[0]
                j = k // array.shape[0]
                average_formatted[i][j] = average[0][k]
            t2 = time.time()

            tt2 = time.time()
            self.logger.info(f"Total time: {tt2-tt1}")
            return average_formatted
        except IndexError:
            tt2 = time.time()
            self.logger.info(f"Total time: {tt2-tt1}")
            return average


    #Equivilant to dCdW3 without jacobian multiplication (5-6x faster!)
    #Jacobian of crossEntropyLoss with respect to W3
    def calculate_dCdW3_quick(self, instance):

        dCdW3 = np.empty(self.W3.shape)
        for i in range(self.W3.shape[0]):
            for j in range(self.W3.shape[1]):
                dCdW3[i][j] = (self.yhat[instance][j]-self.y[instance][j])*self.A3[instance][i]

        return dCdW3


    def calculate_dCdW2(self, instance):
        d1 = self.calculate_dCdyhat(instance)
        d2 = self.calculate_dyhatdZ4(instance)
        d3 = self.calculate_dZ4dA3(instance)
        d4 = self.calculate_dA3dZ3(instance)
        d5 = self.calculate_dZ3dW2(instance)
        
        #dot jacobians together
        dCdZ4 = np.dot(d1, d2)
        dCdA3 = np.dot(dCdZ4, d3)
        dCdZ3 = np.dot(dCdA3, d4)
        dCdW2 = np.dot(dCdZ3, d5)
        
        return dCdW2
    
    
    def calculate_dCdW1(self, instance):
        d1 = self.calculate_dCdyhat(instance)
        d2 = self.calculate_dyhatdZ4(instance)
        d3 = self.calculate_dZ4dA3(instance)
        d4 = self.calculate_dA3dZ3(instance)
        d5 = self.calculate_dZ3dA2(instance)
        d6 = self.calculate_dA2dZ2(instance)
        d7 = self.calculate_dZ2dW1(instance)
        
        #dot jacobians together
        dCdZ4 = np.dot(d1, d2)
        dCdA3 = np.dot(dCdZ4, d3)
        dCdZ3 = np.dot(dCdA3, d4)
        dCdA2 = np.dot(dCdZ3, d5)
        dCdZ2 = np.dot(dCdA2, d6)
        dCdW1 = np.dot(dCdZ2, d7)
        
        return dCdW1
    
    
    def calculate_dCdB3(self, instance):
        dCdB3 = np.empty(self.B3.shape)
        for i in range(self.B3.shape[0]):
            dCdB3[i] = (self.yhat[instance][i]-self.y[instance][i])
                
        return dCdB3
    
    
    def calculate_dCdB2(self, instance):
        d1 = self.calculate_dCdyhat(instance)
        d2 = self.calculate_dyhatdZ4(instance)
        d3 = self.calculate_dZ4dA3(instance)
        d4 = self.calculate_dA3dZ3(instance)
        d5 = self.calculate_dZ3dW2(instance)
        
        #dot jacobians together
        dCdZ4 = np.dot(d1, d2)
        dCdA3 = np.dot(dCdZ4, d3)
        dCdZ3 = np.dot(dCdA3, d4)
        
        return dCdZ3
    
    
    def calculate_dCdB1(self, instance):
        d1 = self.calculate_dCdyhat(instance)
        d2 = self.calculate_dyhatdZ4(instance)
        d3 = self.calculate_dZ4dA3(instance)
        d4 = self.calculate_dA3dZ3(instance)
        d5 = self.calculate_dZ3dA2(instance)
        d6 = self.calculate_dA2dZ2(instance)
        
        #dot jacobians together
        dCdZ4 = np.dot(d1, d2)
        dCdA3 = np.dot(dCdZ4, d3)
        dCdZ3 = np.dot(dCdA3, d4)
        dCdA2 = np.dot(dCdZ3, d5)
        dCdZ2 = np.dot(dCdA2, d6)
        
        return dCdZ2
    
    
    def backPropagation(self, rate=0.01, indices=None):
        self.logger.info("computing gradients")
        self.forward(self.X)
        
        try:
            # calculate gradients
            w3_grad = self.calculate_gradient_average(self.W3, self.calculate_dCdW3, indices=indices)
            b3_grad = self.calculate_gradient_average(self.B3, self.calculate_dCdB3, indices=indices)
            w2_grad = self.calculate_gradient_average(self.W2, self.calculate_dCdW2, indices=indices)
            b2_grad = self.calculate_gradient_average(self.B2, self.calculate_dCdB2, indices=indices)
            w1_grad = self.calculate_gradient_average(self.W1, self.calculate_dCdW1, indices=indices)
            b1_grad = self.calculate_gradient_average(self.B1, self.calculate_dCdB1, indices=indices)
        except ValueError:
            return ValueError
        
        # set underflow values to zero
        w3_grad = np.nan_to_num(w3_grad, nan=0.0)
        b3_grad = np.nan_to_num(b3_grad, nan=0.0)
        w2_grad = np.nan_to_num(w2_grad, nan=0.0)
        b2_grad = np.nan_to_num(b2_grad, nan=0.0)
        w1_grad = np.nan_to_num(w1_grad, nan=0.0)
        b1_grad = np.nan_to_num(b1_grad, nan=0.0)
    
        # update weights and biases
        self.W3 = self.W3 - rate * w3_grad
        self.B3 = self.B3 - rate * b3_grad
        self.W2 = self.W2 - rate * w2_grad
        self.B2 = self.B2 - rate * b2_grad
        self.W1 = self.W1 - rate * w1_grad
        self.B1 = self.B1 - rate * b1_grad
        
        
    def instance_gradientDescent(self, instance):
        self.logger.info("single instance gradient descent has begun")
        learning_rate = .00001
        t1 = time.time()
        prev_epoch = np.mean(self.crossEntropyLoss(self.X_test, self.y_test))
        t2 = time.time()
        self.logger.info(f"Time to calculate initial cross entropy loss: {t2-t1}")
        
        for i in range(50):
            self.classify_instance(self.X, self.y, instance)
            t1 = time.time()
            
            self.logger.info("computing gradients")
            self.W3 = self.W3 - learning_rate * self.dCdW3_quick(self.A3, self.y, instance)
            
            t2 = time.time()
            
            print("epoch:", i, "\n\tTime:", t2-t1)


    def gradientDescent(self):
        print("gradient descent has begun")
        learning_rate = .01
        t1 = time.time()
        prev_epoch = np.mean(self.crossEntropyLoss(self.X_test, self.y_test))
        t2 = time.time()
        print("Time to calculate initial cross entropy loss:", t2-t1)
        
        for i in range(35):
            t1 = time.time()
            
            self.backPropagation(self.X, self.y, rate=learning_rate)
            cur_epoch = np.mean(self.crossEntropyLoss(self.X_test, self.y_test)[0])
            
            t2 = time.time()
            
            print("epoch:", i, "\n\tcross entropy loss:", cur_epoch, "\n\tratio:", cur_epoch/prev_epoch, "\n\tTime:", t2-t1)
            prev_epoch = cur_epoch
         
            
    def batchGradientDescent(self, X_in, y_in, learning_rate=.001, batch_size=10):
        print("batch gradient descent has begun")
        
        #make copy of parameters
        X = np.array(X_in, copy=True)
        y = np.array(y_in, copy=True)
        
        indices = np.random.randint(len(X), size=batch_size)
        X_batch = X[indices,:]
        y_batch = y[indices,:]
        
        t1 = time.time()
        initial_epoch = np.mean(self.crossEntropyLoss(self.X_test, self.y_test))
        prev_epoch = initial_epoch
        t2 = time.time()
        print("Time to calculate initial cross entropy loss:", t2-t1)
        
        for i in range(100):
            t1 = time.time()
            
            indices = np.random.randint(len(X), size=batch_size)
            X_batch = X[indices,:]
            y_batch = y[indices,:]
            
            self.backPropagation(X_batch, y_batch, rate=learning_rate)
            cur_epoch = np.mean(self.crossEntropyLoss(self.X_test, self.y_test)[0])
            
            t2 = time.time()
            
            print("epoch:", i, "\n\tcross entropy loss:", cur_epoch, "\n\tratio:", cur_epoch/prev_epoch, "\n\tTime:", t2-t1)
            prev_epoch = cur_epoch
            
        return
        group_size = .1
        learning_rate = 0.00001
        prev_epoch = self.crossEntropyLoss(self.X_test, self.y_test)
        
        for i in range(10):
            #stratified shuffle split
            # here
            
            self.backPropagation(X, y, rate=learning_rate)
            
            
            cur_epoch = self.crossEntropyLoss(self.X_test, self.y_test)
            print("epoch:", i, "cross entropy loss:", cur_epoch, "ratio:", cur_epoch/prev_epoch)
            prev_epoch = cur_epoch
       
            
    def classify_set(self, X, y):
        prediction = self.forward_test(X)
        for i in range(len(X)):
            print(prediction[i])
            print(y[i])
            print("\n")
            
        return prediction
            
    
    def classify_instance(self, X, y, instance):
        prediction = self.forward_test([X[instance]])
        print("Predicted:", prediction[0])
        print("Actual:", y[instance])
        
        return prediction[0]
            
    

    
