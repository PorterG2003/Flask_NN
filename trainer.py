#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 14 19:43:39 2023

@author: porter
"""

from mlxtend.data import loadlocal_mnist
import platform
import numpy as np
from nn import NeuralNet as NN
import time
from random import randint
import logging

class Trainer:
    def __init__(self, nn=None, logger=None):
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

        self.stop = False
        self.running = False
        if not nn:
            self.logger.info("Loading Data")
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

            self.logger.info("Formatting Data")
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

            self.nn = NN(train_X, train_y_formatted, test_X, test_y_formatted)
        else:
            self.nn = nn

    def stop_training(self):
        self.stop = True
        return "Stopped"

    def SGD(self, batch_size=256, learning_rate=0.01):
        self.running = True
        self.logger.info("\nBeginning Stochastic Gradient Descent...")
        self.logger.info(f"Learning Rate: {learning_rate}")
        nn = self.nn
        nn.save()

        #make copy of parameters
        X = np.array(nn.X, copy=True)
        y = np.array(nn.y, copy=True)

        indices = np.random.randint(len(X), size=batch_size)

        t1 = time.time()
        initial_epoch = np.mean(nn.crossEntropyLoss())
        prev_epoch = initial_epoch
        cur_epoch = initial_epoch
        top_cel = initial_epoch
        t2 = time.time()
        self.logger.info(f"Time to calculate initial cross entropy loss: {t2-t1}")

        for i in range(3600):
            if self.stop:
                self.running = False
                self.stop = False
                self.logger.info("Stopped Training")
                return

            t1 = time.time()
            if nn.backPropagation(rate=learning_rate, indices=indices) == ValueError:
                cur_epoch = np.mean(nn.crossEntropyLoss())
                if cur_epoch < prev_epoch:
                    top_cel = cur_epoch
                nn.save(notes=f"SGD Value error: \n epoch: {i} cross entropy loss: {cur_epoch} \n best cross entropy loss: {top_cel}")
                return

            t2 = time.time()
            cur_epoch = np.mean(nn.crossEntropyLoss())
            if cur_epoch < prev_epoch:
                top_cel = cur_epoch
            indices = np.random.randint(len(X), size=batch_size)

            self.logger.info(f"epoch: {i}\n\tcross entropy loss: {cur_epoch}\n\tratio: {cur_epoch/prev_epoch}\n\tTime: {t2-t1}")
            prev_epoch = cur_epoch

        #nn.save()


if __name__ == '__main__':
    trainer = Trainer()
    trainer.SGD()
