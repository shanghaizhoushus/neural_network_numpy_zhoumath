# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 23:55:42 2024

@author: zhoushus
"""

#Import packages
import numpy as np
from sklearn.metrics import roc_auc_score

#Earlystopper Class
class Earlystopper:
    def __init__(self, early_stop_rounds, n_iters, decay_rounds, verbose):
        """
        Initialize the Earlystopper with early stopping and learning rate decay settings.
        :param early_stop_rounds: Number of rounds to wait for improvement before stopping.
        :param n_iters: Total number of training iterations.
        :param decay_rounds: Number of rounds before triggering learning rate decay.
        :param verbose: Verbosity level for logging progress.
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
        :param model: The model being trained.
        :param X_train: Training features.
        :param y_train: Training labels.
        :param X_val: Validation features.
        :param y_val: Validation labels.
        :return: Cached linear weights and bias if early stopping is triggered, otherwise None.
        """
        self.current_round += 1
        self.current_early_stop_rounds += 1
        self.current_decay_rounds += 1
        train_logodds = model._forward(X_train)
        train_predicts = 1 / (1 + np.exp(-train_logodds))
        train_auc = roc_auc_score(y_train, train_predicts)
        val_logodds = model._forward(X_val)
        val_predicts = 1 / (1 + np.exp(-val_logodds))
        val_auc = roc_auc_score(y_val, val_predicts)
        
        if self.verbose:
            if (self.current_round % self.verbose) == 0:
                print(f"Current round: {self.current_round}\t"
                      f"Train AUC: {train_auc:.4f}\t"
                      f"Val AUC: {val_auc:.4f}")
        
        if val_auc > self.best_auc:
            self.best_auc = val_auc
            self.linear_cache = model.linear
            self.bias_cache = model.bias
            self.current_early_stop_rounds = 0
            self.current_decay_rounds = 0
        
        if self.decay_rounds is not None:
            if self.current_decay_rounds >= self.decay_rounds:
                model.learning_rate *= 0.1
                self.current_decay_rounds = 0
                if self.verbose:
                    print(f"Learning rate decay triggered at round: {self.current_round}")
        
        if self.current_early_stop_rounds >= self.early_stop_rounds or self.current_round >= self.n_iters:
            if self.verbose:
                print(f"Stop triggered at round: {self.current_round}\t"
                      f"Best Val AUC: {self.best_auc:.4f}\t")
                if self.current_round >= self.n_iters:
                    print(f"Best round: {self.current_round}")
                else:
                    print(f"Best round: {self.current_round - self.early_stop_rounds}")
            
            return self.linear_cache, self.bias_cache
        
        return None, None

#AdamOptimizer Class
class AdamOptimizer:
    def __init__(self, beta1, beta2, shape):
        """
        Initialize the Adam optimizer with hyperparameters and initial values.
        :param beta1: Exponential decay rate for the first moment estimates.
        :param beta2: Exponential decay rate for the second moment estimates.
        :param shape: Shape of the weights to be optimized.
        """
        self.beta1 = beta1
        self.beta2 = beta2
        self.m = np.zeros(shape)
        self.v = np.zeros(shape)
        self.t = 0
    
    def _renew(self, gradient):
        """
        Update the parameters using the gradient provided.
        :param gradient: Gradient to be used for updating weights.
        :return: Updated parameter values after applying the Adam update rule.
        """
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
        self.v = self.beta2 * self.v + (1 - self.beta2) * np.square(gradient)
        adjusted_m = self.m / (1 - self.beta1 ** self.t)
        adjusted_v = self.v / (1 - self.beta2 ** self.t)
        return adjusted_m / (np.sqrt(adjusted_v) + 1e-8)
