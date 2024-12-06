# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 20:26:28 2024

@author: zhoushus
"""

#Import packages
import numpy as np
from neural_network_helpers_zhoumath import Earlystopper, AdamOptimizer

# LogisticRegressionZhoumath Class
class LogisticRegressionZhoumath:
    def __init__(self, learning_rate, n_iters, beta1=0.9, beta2=0.999, batch_size=256):
        """
        Initialize the Logistic Regression model with learning rate, iterations, and optimizer settings.
        :param learning_rate: Learning rate for training.
        :param n_iters: Number of iterations for training.
        :param beta1: Exponential decay rate for the first moment estimates in the Adam optimizer.
        :param beta2: Exponential decay rate for the second moment estimates in the Adam optimizer.
        :param batch_size: Size of the batch used in training.
        """
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.beta1 = beta1
        self.beta2 = beta2
        self.batch_size = batch_size
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, early_stop_rounds=None,
            decay_rounds=None, verbose=False, random_seed=42):
        """
        Fit the Logistic Regression model to the training data.
        :param X_train: Training features.
        :param y_train: Training labels.
        :param X_val: Validation features for early stopping.
        :param y_val: Validation labels for early stopping.
        :param early_stop_rounds: Number of rounds for early stopping.
        :param decay_rounds: Number of rounds before learning rate decay.
        :param verbose: Verbosity level for logging progress.
        :param random_seed: Random seed for reproducibility.
        """
        np.random.seed(random_seed)
        early_stopper = None
        X_train = LogisticRegressionZhoumath._standardlize(X_train)
        y_train = np.ascontiguousarray(y_train).reshape(-1, 1)
        self._init_weights(X_train)
        
        if X_val is not None and y_val is not None and early_stop_rounds is not None:
            X_val = LogisticRegressionZhoumath._standardlize(X_val)
            y_val = np.ascontiguousarray(y_val).reshape(-1, 1)
            early_stopper = Earlystopper(early_stop_rounds, self.n_iters, decay_rounds, verbose)
            
        if self.batch_size is not None:
            self.num_batches = (X_train.shape[0] // self.batch_size + 1)
            self.current_batch = 0
        
        for i in range(self.n_iters):
            logodds = self._forward(X_train)
            predicts = 1 / (1 + np.exp(-logodds))
            residuals = predicts - y_train
            
            if self.batch_size is None:
                self._backward(X_train, residuals)
            else:
                X_train_batch, residuals_batch = self._get_batch(X_train, residuals)
                self._backward(X_train_batch, residuals_batch)
            
            if early_stopper is not None:
                linear_cache, bias_cache = early_stopper._evaluate_early_stop(self, X_train, y_train, X_val, y_val)
                if linear_cache is not None:
                    self.linear = linear_cache
                    self.bias = bias_cache
                    return
    
    def _init_weights(self, X):
        """
        Initialize model weights using a normal distribution.
        :param X: Training data features used to determine the number of features.
        """
        self.num_features = X.shape[1]
        self.linear = np.random.normal(0, 2 / self.num_features, (self.num_features, 1))
        self.linear_optimizer = AdamOptimizer(self.beta1, self.beta2, self.linear.shape)
        self.bias = np.zeros((1))
        self.bias_optimizer = AdamOptimizer(self.beta1, self.beta2, self.bias.shape)
    
    @staticmethod
    def _standardlize(X):
        """
        Standardize the dataset by removing the mean and scaling to unit variance.
        :param X: Input feature matrix.
        :return: Standardized feature matrix.
        """
        X = (X - X.mean(axis = 0)) / X.std(axis = 0)
        return np.ascontiguousarray(X)
    
    def _get_batch(self, X_train, residuals):
        """
        Retrieve the next batch of training data and residuals for mini-batch training.
        :param X_train: Full training feature matrix.
        :param residuals: Residuals from the predictions of the full training data.
        :return: Batch of training features and corresponding residuals.
        """
        if self.current_batch == self.num_batches:
            X_train_batch = np.ascontiguousarray(X_train[self.current_batch*self.batch_size:, :].copy())
            residuals_batch = np.ascontiguousarray(residuals[self.current_batch*self.batch_size:, :].copy())
            self.current_batch = 0
        else:
            X_train_batch = X_train[self.current_batch*self.batch_size:(self.current_batch+1)*self.batch_size, :].copy()
            X_train_batch = np.ascontiguousarray(X_train_batch)
            residuals_batch = residuals[self.current_batch*self.batch_size:(self.current_batch+1)*self.batch_size, :].copy()
            residuals_batch = np.ascontiguousarray(residuals_batch)
            self.current_batch += 1
            
        return X_train_batch, residuals_batch
    
    def _forward(self, X_train):
        """
        Perform the forward pass to calculate log-odds for the given features.
        :param X_train: Input feature matrix.
        :return: Log-odds for each sample.
        """
        logodds = np.dot(X_train, self.linear) + self.bias
        return logodds
    
    def _backward(self, X_train, residuals):
        """
        Perform the backward pass to update model weights.
        :param X_train: Input feature matrix.
        :param residuals: Residuals between predicted and actual labels.
        """
        linear_grad = np.dot(np.ascontiguousarray(X_train.T), residuals)
        linear_delta = self.linear_optimizer._renew(linear_grad)
        self.linear -= self.learning_rate * linear_delta
        bias_grad = np.dot(np.ones((1, residuals.shape[0])), residuals).reshape(-1)
        bias_delta = self.bias_optimizer._renew(bias_grad)
        self.bias -=  self.learning_rate * bias_delta
    
    def predict(self, X_test):
        """
        Make predictions for the given test data.
        :param X_test: Test feature matrix.
        :return: Probability predictions for each sample belonging to each class.
        """
        X_test = LogisticRegressionZhoumath._standardlize(X_test)
        logodds = self._forward(X_test)
        predicts = 1 / (1 + np.exp(-logodds))
        return np.hstack([1 - predicts, predicts])
