#-*- coding: utf-8 -*-
import numpy as np
import math


def hello():
    print('Hello from soft_margin_svm.py')


def svm_train_bgd(X: np.ndarray, y: np.ndarray, num_epochs: int=100, C: float=5.0, eta: float=0.001):
  
    # Implement your algorithm and return state (e.g., learned model)
    num_data, num_features = X.shape
    
    np.random.seed(0)
    W = np.zeros((1, num_features), dtype=X.dtype)
    b = np.zeros((1), dtype=X.dtype)
    
    for j in range(1, num_epochs+1):
        W_grad = W.transpose() - C * np.matmul(X.transpose(), y*np.where(y*(np.matmul(X,W.transpose())+b)<1,1,0))
        b_grad = -C * np.matmul(np.where(y*(np.matmul(X,W.transpose())+b)<1,1,0).transpose(), y)
        W =  W - eta * W_grad.transpose()
        b = b - eta * b_grad
    b = b.reshape((1))
        
    return W, b


def svm_test(W: np.ndarray, b: np.ndarray, X: np.ndarray, y: np.ndarray):
    
    pred = (X @ W.T + b[np.newaxis, :] > 0).astype(y.dtype)*2 - 1
    accuracy = np.mean((pred == y).astype(np.float32))
    return accuracy
