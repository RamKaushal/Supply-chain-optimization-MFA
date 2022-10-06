import os
from unicodedata import east_asian_width
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class perceptron:
    def __init__(self,eta,epoch):
        self.weights = np.random.rand(3) * 1e-4
        self.eta = eta 
        self.epoch = epoch 
    def fit(self,X,y):
        self.X = X
        self.y = y 
        X_with_bias = np.c_[self.X,np.ones(4,1)]
        print("X with bias is: ".format(X_with_bias))
        return X_with_bias


if __name__ == "__main__":
    