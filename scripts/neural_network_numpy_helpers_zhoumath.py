# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 23:55:42 2024

@author: zhoushus
"""

#Import packages
import numpy as np
from sklearn.metrics import roc_auc_score

class Earlystopper:
    """
    A class to implement early stopping during model training, including learning rate decay.
    
    Attributes:
    -----------
    early_stop_rounds : int
        The number of rounds to wait for improvement before stopping early.
    n_iters : int
        Total number of training iterations.
    decay_rounds : int
        The number of rounds before triggering learning rate decay.
    verbose : bool
        Whether to log the training progress.
    current_round : int
        The current training round.
    current_early_stop_rounds : int
        The number of rounds since the last improvement.
    current_decay_rounds : int
        The number of rounds since the last learning rate decay.
    best_auc : float
        The best validation AUC score encountered during training.
    linears_cache : list of ndarray
        Cached weights of the model when early stopping is triggered.
    biases_cache : list of ndarray
        Cached biases of the model when early stopping is triggered.
    
    Methods:
    --------
    _evaluate_early_stop(model, X_train, y_train, X_val, y_val)
        Evaluates the model's performance on the training and validation sets,
        triggering early stopping or learning rate decay if necessary.
    """
    def __init__(self, early_stop_rounds, n_iters, decay_rounds, verbose):
        """
        Initialize the Earlystopper with early stopping and learning rate decay settings.
        
        :param early_stop_rounds: int, the number of rounds to wait for improvement before stopping.
        :param n_iters: int, the total number of training iterations.
        :param decay_rounds: int, the number of rounds before triggering learning rate decay.
        :param verbose: bool, whether to print progress during training.
        """
        self.early_stop_rounds = early_stop_rounds
        self.n_iters = n_iters
        self.decay_rounds = decay_rounds
        self.verbose = verbose
        self.current_round = 0
        self.current_early_stop_rounds = 0
        self.current_decay_rounds = 0
        self.best_auc = -np.inf
    
    def _evaluate_early_stop(self, model, X_train, y_train, X_val, y_val):
        """
        Evaluate model performance and determine whether to apply early stopping or learning rate decay.
        
        :param model: object, the model being trained.
        :param X_train: ndarray, training features.
        :param y_train: ndarray, training labels.
        :param X_val: ndarray, validation features.
        :param y_val: ndarray, validation labels.
        :return: tuple of (linears_cache, biases_cache), where linears_cache and biases_cache are 
                 the cached weights and biases of the model if early stopping is triggered. 
                 Returns None, None if no action is taken.
        """
        self.current_round += 1
        self.current_early_stop_rounds += 1
        self.current_decay_rounds += 1
        _, _, _, train_logodds = model._forward(X_train, 0)
        train_predicts = 1 / (1 + np.exp(-train_logodds))
        train_auc = roc_auc_score(y_train, train_predicts)
        _, _, _, val_logodds = model._forward(X_val, 0)
        val_predicts = 1 / (1 + np.exp(-val_logodds))
        val_auc = roc_auc_score(y_val, val_predicts)
        
        if (self.verbose) and (self.current_round % self.verbose) == 0:
            print(f"Current round: {self.current_round}\t"
                  f"Train AUC: {train_auc:.4f}\t"
                  f"Val AUC: {val_auc:.4f}")
        
        if val_auc > self.best_auc:
            self.best_auc = val_auc
            self.linears_cache = model.linears
            self.biases_cache = model.biases
            self.current_early_stop_rounds = 0
            self.current_decay_rounds = 0
        
        if (self.decay_rounds is not None) and (self.current_decay_rounds >= self.decay_rounds):
            model.learning_rate *= 0.1
            self.current_decay_rounds = 0
            
            if self.verbose:
                print(f"Learning rate decay triggered at round: {self.current_round}")
            
        
        if (self.current_early_stop_rounds >= self.early_stop_rounds) or (self.current_round >= self.n_iters):
            
            if self.verbose:
                print(f"Stop triggered at round: {self.current_round}\t"
                      f"Best Val AUC: {self.best_auc:.4f}\t")
                
                if self.current_round >= self.n_iters:
                    print(f"Best round: {self.current_round}")
                else:
                    print(f"Best round: {self.current_round - self.early_stop_rounds}")
                
            
            return self.linears_cache, self.biases_cache
        
        return None, None
    

class AdamOptimizer:
    """
    A class to implement the Adam optimizer for gradient-based optimization of the model's parameters.

    Attributes:
    -----------
    beta1 : float
        Exponential decay rate for the first moment estimate.
    beta2 : float
        Exponential decay rate for the second moment estimate.
    m : ndarray
        The running average of the gradient.
    v : ndarray
        The running average of the squared gradient.
    t : int
        Time step, used to correct the bias in the moment estimates.

    Methods:
    --------
    _renew(gradient)
        Updates the model parameters using the Adam update rule based on the provided gradient.
    """
    def __init__(self, beta1, beta2, shape):
        """
        Initialize the Adam optimizer with hyperparameters and initial values.
        
        :param beta1: float, the exponential decay rate for the first moment estimate.
        :param beta2: float, the exponential decay rate for the second moment estimate.
        :param shape: tuple, the shape of the model parameters (weights) to be optimized.
        """
        self.beta1 = beta1
        self.beta2 = beta2
        self.m = np.zeros(shape)
        self.v = np.zeros(shape)
        self.t = 0
    
    def _renew(self, gradient):
        """
        Update the model parameters using the provided gradient, according to the Adam update rule.
        
        :param gradient: ndarray, the gradient of the objective function with respect to the parameters.
        :return: ndarray, the updated parameter values after applying the Adam update rule.
        """
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
        self.v = self.beta2 * self.v + (1 - self.beta2) * np.square(gradient)
        adjusted_m = self.m / (1 - self.beta1 ** self.t)
        adjusted_v = self.v / (1 - self.beta2 ** self.t)
        return adjusted_m / (np.sqrt(adjusted_v) + 1e-8)
