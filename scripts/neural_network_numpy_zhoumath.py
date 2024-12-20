# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 15:57:45 2024

@author: zhoushus
"""

#Import packages
import pickle
import numpy as np
from neural_network_numpy_helpers_zhoumath import Earlystopper, AdamOptimizer

class NeuralNetworkNumpyZhoumath:
    """
    A neural network class that implements a fully connected feedforward neural network
    with ReLU activation for hidden layers and sigmoid activation for the output layer.
    The model uses the Adam optimizer for weight updates and supports both batch gradient descent
    and mini-batch gradient descent.

    Attributes:
    -----------
    learning_rate : float
        The learning rate used for training.
    n_iters : int
        The number of iterations (epochs) to train the model.
    dim_hiddens : list of int
        A list representing the number of neurons in each hidden layer.
    dropout_rate: float
        The percentage of dropout of linear matries.
    beta1 : float
        Exponential decay rate for the first moment estimate in the Adam optimizer.
    beta2 : float
        Exponential decay rate for the second moment estimate in the Adam optimizer.
    batch_size : int or None
        The size of the mini-batches used in training. If None, full-batch training is used.
    linears : list of ndarray
        The weights for each layer.
    biases : list of ndarray
        The biases for each layer.
    linears_gradients : list of ndarray
        The gradients for the weights.
    biases_gradients : list of ndarray
        The gradients for the biases.
    linears_optimizers : list of AdamOptimizer
        The Adam optimizers used for each layer's weights.
    biases_optimizers : list of AdamOptimizer
        The Adam optimizers used for each layer's biases.

    Methods:
    --------
    fit(X_train, y_train, X_val=None, y_val=None, early_stop_rounds=None,
        decay_rounds=None, verbose=False, random_seed=42)
        Trains the neural network on the given training data with optional early stopping and learning rate decay.
    _standardlize(X)
        Standardizes the input data by removing the mean and scaling to unit variance.
    _init_weights(X)
        Initializes the weights and biases for the neural network layers.
    _get_batch(X_train, y_train)
        Retrieves the next mini-batch of training data for batch or mini-batch gradient descent.
    _forward(self, X_train, dropout_rate, random_seed = 42)
        Performs the forward pass to calculate the log-odds and activations for each layer.
    _backward(inputs, inputs_gradients, residuals)
        Performs the backward pass to compute gradients and update the weights and biases.
    _remove_grads(inputs, inputs_gradients, residuals)
        Remove the gradients when train ends.
    predict(X_test)
        Makes predictions for the given test data by calculating the class probabilities.
    to_pkl(model_name)
        Save the model object to a pickle file.
    """
    def __init__(self, learning_rate, n_iters, dim_hiddens, dropout_rate=0,
                 beta1=0.9, beta2=0.999, batch_size=None):
        """
        Initialize the neural network with the given parameters.
        
        :param learning_rate: float, learning rate used for training the model.
        :param n_iters: int, the number of iterations (epochs) to train the model.
        :param dim_hiddens: list of int, the number of neurons in each hidden layer.
        :param dropout_rate: float, The percentage of dropout of linear matries.
        :param beta1: float, the exponential decay rate for the first moment estimate in Adam optimizer.
        :param beta2: float, the exponential decay rate for the second moment estimate in Adam optimizer.
        :param batch_size: int or None, the size of each mini-batch.
        If None, the model will use full-batch training.
        """
        if (dropout_rate < 0) or (dropout_rate >= 1):
            raise ValueError("Dropout rate should be in [0,1).")
        
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.dim_hiddens = dim_hiddens
        self.dropout_rate = dropout_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.batch_size = batch_size
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, early_stop_rounds=None,
            decay_rounds=None, verbose=False, random_seed=42):
        """
        Train the neural network model on the given training data.
        
        :param X_train: ndarray, shape (n_samples, n_features), training feature matrix.
        :param y_train: ndarray, shape (n_samples, 1), training labels.
        :param X_val: ndarray, shape (n_samples, n_features), optional,
        validation feature matrix for early stopping.
        :param y_val: ndarray, shape (n_samples, 1), optional, validation labels for early stopping.
        :param early_stop_rounds: int, optional, the number of rounds without improvement before stopping early.
        :param decay_rounds: int, optional, the number of rounds before decaying the learning rate.
        :param verbose: bool, whether to print training progress at each iteration (default is False).
        :param random_seed: int, random seed for reproducibility (default is 42).
        """
        np.random.seed(random_seed)
        early_stopper = None
        X_train = NeuralNetworkNumpyZhoumath._standardlize(X_train)
        y_train = np.ascontiguousarray(y_train).reshape(-1, 1)
        self._init_weights(X_train)
        
        if (X_val is not None) and (y_val is not None) and (early_stop_rounds is not None):
            X_val = NeuralNetworkNumpyZhoumath._standardlize(X_val)
            y_val = np.ascontiguousarray(y_val).reshape(-1, 1)
            early_stopper = Earlystopper(early_stop_rounds, self.n_iters, decay_rounds, verbose)
        
        if self.batch_size is not None:
            self.num_batches = X_train.shape[0] // self.batch_size + 1
            self.current_batch = 0
        
        for i in range(self.n_iters):
            
            if self.batch_size is None:
                inputs, inputs_gradients, dropout_locations, logodds = self._forward(X_train, self.dropout_rate,
                                                                                     random_seed+i)
                predicts = 1 / (1 + np.exp(-logodds))
                residuals = predicts - y_train
                self._backward(inputs, inputs_gradients, dropout_locations, residuals)
            else:
                X_train_batch, y_train_batch = self._get_batch(X_train, y_train)
                inputs, inputs_gradients, dropout_locations, logodds = self._forward(X_train_batch, self.dropout_rate,
                                                                                     random_seed+i)
                predicts = 1 / (1 + np.exp(-logodds))
                residuals = predicts - y_train_batch
                self._backward(inputs, inputs_gradients, dropout_locations, residuals)
            
            if early_stopper is not None:
                linears_cache, biases_cache = early_stopper._evaluate_early_stop(self, X_train, y_train,
                                                                                 X_val, y_val)
                
                if (linears_cache is not None) and (biases_cache is not None):
                    self.linears = linears_cache
                    self.biaes = biases_cache
                    self._remove_grads()
                    return
                
            
        
        self._remove_grads()
    
    @staticmethod
    def _standardlize(X):
        """
        Standardize the input data by removing the mean and scaling to unit variance.
        
        :param X: ndarray, shape (n_samples, n_features), the input feature matrix to standardize.
        :return: ndarray, shape (n_samples, n_features), the standardized feature matrix.
        """
        X = (X - X.mean(axis = 0)) / X.std(axis = 0)
        return np.ascontiguousarray(X)
    
    def _init_weights(self, X):
        """
        Initialize the model weights and biases using kaiming initialization.
        
        :param X: ndarray, shape (n_samples, n_features),
        the input feature matrix to determine the input dimension.
        """
        self.dims = [X.shape[1]] + self.dim_hiddens + [1]
        self.linears = []
        self.linears_gradients = []
        self.linears_optimizers = []
        self.biases = []
        self.biases_gradients = []
        self.biases_optimizers = []
        
        for i in range(len(self.dims) - 1):
            self.linears.append(np.random.normal(0, 2/self.dims[i], (self.dims[i], self.dims[i+1])))
            self.linears_gradients.append(np.zeros((self.dims[i], self.dims[i+1])))
            self.linears_optimizers.append(AdamOptimizer(self.beta1, self.beta2, (self.dims[i], self.dims[i+1])))
            self.biases.append(np.zeros((1, self.dims[i+1])))
            self.biases_gradients.append(np.zeros((1, self.dims[i+1])))
            self.biases_optimizers.append(AdamOptimizer(self.beta1, self.beta2, (1, self.dims[i+1])))
        
    
    def _get_batch(self, X_train, y_train):
        """
        Retrieve the next batch of training data and residuals for mini-batch training.
        
        :param X_train: ndarray, shape (n_samples, n_features), Full training feature matrix.
        :param y_train: ndarray, shape (n_samples, 1), Full training label matrix.
        :return: tuple of (X_train_batch, y_train_batch), where:
                 X_train_batch: ndarray, shape (batch_size, n_features), training data of the batch,
                 y_train_batch: ndarray, shape (batch_size, 1), labels of the batch.
        """
        if self.current_batch == self.num_batches:
            X_train_batch = np.ascontiguousarray(X_train[self.current_batch*self.batch_size:, :].copy())
            y_train_batch = np.ascontiguousarray(y_train[self.current_batch*self.batch_size:, :].copy())
            self.current_batch = 0
        else:
            X_train_batch = X_train[self.current_batch*self.batch_size:(self.current_batch+1)*self.batch_size, :].copy()
            X_train_batch = np.ascontiguousarray(X_train_batch)
            y_train_batch = y_train[self.current_batch*self.batch_size:(self.current_batch+1)*self.batch_size, :].copy()
            y_train_batch = np.ascontiguousarray(y_train_batch)
            self.current_batch += 1
        
        return X_train_batch, y_train_batch
    
    def _forward(self, X_train, dropout_rate, random_seed = 42):
        """
        Perform the forward pass through the network to compute log-odds.
        
        :param X_train: ndarray, shape (n_samples, n_features), the input feature matrix.
        :param dropout_rate: int, the rate of dropout in forward propagation.
        :param random_seed: int, random seed for reproducibility (default is 42).
        :return: tuple of (inputs, inputs_gradients, logodds), where:
                 inputs: list of ndarray, the list of activations for each layer,
                 inputs_gradients: list of ndarray, the list of gradients for each layer,
                 dropout_locations: list of ndarray, the list of dropout locations for each layer,
                 logodds: ndarray, shape (batch_size, 1), the final log-odds values for the output layer.
        """
        np.random.seed(random_seed)
        inputs = [X_train]
        inputs_gradients = [np.zeros((X_train.shape))]
        dropout_locations = [np.zeros((X_train.shape))]
        
        for i in range(len(self.dims) - 1):
            mask = np.random.uniform(size = self.linears[i].shape) > 1 - dropout_rate
            dropout_locations.append(mask)
            linear_i = self.linears[i].copy()
            linear_i[mask] = 0
            linear_i *= 1 / (1 - dropout_rate)
            
            if i < len(self.dims) - 2:
                result = np.dot(inputs[i], linear_i) + self.biases[i]
                inputs.append(np.where(result > 0, result, 0))
                inputs_gradients.append(np.zeros((self.dims[i], self.dims[i+1])))
            else:
                logodds = np.dot(inputs[i], linear_i) + self.biases[i]
            
        return inputs, inputs_gradients, dropout_locations, logodds
    
    def _backward(self, inputs, inputs_gradients, dropout_locations, residuals):
        """
        Perform the backward pass to update weights and biases based on gradients.
        
        :param inputs: list of ndarray, intermediate activations from the forward pass.
        :param inputs_gradients: list of ndarray, gradients of activations for each layer.
        :param dropout_locations: list of ndarray, the list of dropout locations for each layer
        :param residuals: ndarray, shape (n_samples, 1), the residuals (predictions - true labels).
        """
        inputs_gradients.append(residuals)
        
        for i in range(len(self.dims) - 1)[::-1]:
            self.linears_gradients[i] = np.dot(np.ascontiguousarray(inputs[i].T), inputs_gradients[i+1])
            
            if self.dropout_rate > 0:
                self.linears_gradients[i][dropout_locations[i+1]] = 0
            
            self.linears[i] -= self.learning_rate * self.linears_optimizers[i]._renew(self.linears_gradients[i])
            self.biases_gradients[i] = np.dot(np.ones((1, inputs_gradients[i+1].shape[0])), inputs_gradients[i+1])
            self.biases[i] -= self.learning_rate * self.biases_optimizers[i]._renew(self.biases_gradients[i])
            inputs_gradients[i] = np.dot(inputs_gradients[i+1], np.ascontiguousarray(self.linears[i].T))
            inputs_gradients[i] = inputs_gradients[i] * (inputs[i] > 0).astype(np.float64)
        
    
    def _remove_grads(self):
        """
        Remove the gradients of the model when train ends to reduce the size of the model.
        """
        (self.linears_gradients, self.biases_gradients,
         self.linears_optimizers, self.biases_optimizers, self.linears_gradients) = tuple([None] * 5)
    
    def predict(self, X_test):
        """
        Predict the class probabilities for the given test data.
        
        :param X_test: ndarray, shape (n_samples, n_features), the test feature matrix.
        :return: ndarray, shape (n_samples, 2),
        predicted class probabilities for each sample (negative, positive).
        """
        X_test = NeuralNetworkNumpyZhoumath._standardlize(X_test)
        _, _, _, logodds = self._forward(X_test, 0)
        predicts = 1 / (1 + np.exp(-logodds))
        return np.hstack([1-predicts, predicts])
    
    def to_pkl(self, model_name):
        """
        Save the model object to a pickle file.
        
        :param model_name: String, the name of the pickle file.
        """
        with open(model_name, "wb") as f:
            pickle.dump(self, f)
        
    
