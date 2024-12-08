# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 17:31:11 2024

@author: zhoushus
"""

#Import packages
import sys
import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
script_dir = os.path.abspath(os.path.join(os.getcwd(), '../scripts'))
sys.path.insert(0, script_dir)
from neural_network_numpy_zhoumath import NeuralNetworkNumpyZhoumath
np.random.seed(42)

# Setting
warnings.filterwarnings("ignore", category=UserWarning)

# Load dataset
data = load_breast_cancer(as_frame=True)
X, y = data.data, data.target
'''
data = pd.read_csv('../../HIGGS.csv', header = None, nrows = 110000)
X = data.iloc[:, 1:]
y = data.iloc[:, 0]
'''

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Fit model
learning_rate = 0.001
n_iters = 10000
batch_size = 256
dim_hiddens = [256, 128, 32]
neural_network_model = NeuralNetworkNumpyZhoumath(learning_rate=learning_rate,
                                                  n_iters=n_iters,
                                                  dim_hiddens=dim_hiddens,
                                                  batch_size=batch_size)
tic = time.time()
neural_network_model.fit(X_train=X_train,
                         y_train=y_train,
                         X_val=X_val,
                         y_val=y_val,
                         early_stop_rounds = 100,
                         decay_rounds = 10,
                         verbose=50)
toc = time.time()
gap = toc-tic
print(f'The neural-network-zhoumath-with-null-zhoumath model is bulit in {gap:.4f} seconds.')

# Predict
X_test = np.array(X_test)
tic = time.time()
y_test_pred = neural_network_model.predict(X_test)[:, 1]
toc = time.time()
gap = toc-tic
print(f'The neural-network-zhoumath model is predicted in {gap:.4f} seconds.')
auc_score = roc_auc_score(y_test, y_test_pred)
fpr, tpr, _ = roc_curve(y_test, y_test_pred)
ks = tpr[abs(tpr - fpr).argmax()] - fpr[abs(tpr - fpr).argmax()]
print(f"KS = {ks:.4f}\nAUC = {auc_score:.4f}")

# Plot ROC Curve
plt.plot(fpr, fpr, label="Random Guess")
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc_score:.4f})")
plt.plot(
    [fpr[abs(tpr - fpr).argmax()]] * len(fpr),
    np.linspace(fpr[abs(tpr - fpr).argmax()], tpr[abs(tpr - fpr).argmax()], len(fpr)),
    "--",
)
plt.title("ROC Curve of the Neural Network for test set")
plt.legend()
plt.show()

# Plot the histogram of the weights of the model
for i in range(len(neural_network_model.linears)):
    linear = neural_network_model.linears[i].flatten()
    plt.hist(linear, bins = int(linear.shape[0])//100 + 10)
    plt.title(f"The histogram of the weights in linear {i} of the model")
    plt.show()
    bias = neural_network_model.biases[i].flatten()
    plt.hist(bias, bins = int(bias.shape[0])//100 + 10)
    plt.title(f"The histogram of the weights in bias {i} of the model")
    plt.show()

# Save model to pkl
neural_network_model.to_pkl("neural_network_model_zhoumath.pkl")
