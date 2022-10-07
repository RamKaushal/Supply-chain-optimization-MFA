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
    
    def weights_bias(self,weights,bias):
        return np.dot(weights,bias)
    
    def activation(z):
        return np.where(z>0,1,0)



    def fit(self,X,y):
        self.X = X
        self.y = y 
        X_with_bias = np.c_[self.X,np.ones(4,1)]
        print("X with bias is: ".format(X_with_bias))
        #return X_with_bias
        for i in range(self.epoch):
            z = weights_bias(X_with_bias,self.weights)
            print("Ephoc {} weights+vales is {}".format(i,z))
            y_hat = activation(z)
            print("yhat of epoch {} is {}".format(i,y_hat))
            self.error = self.y - y_hat
            self.weights = self.weights+self.epoch * np.dot(X_with_bias.T,self.error)
    
    def predict(self,x):
        x_with_bias =  np.c_[x,np.ones(4,1)]
        weight_update = weight_bias(x_with_bias,self.weights)
        return activation(weight_update)

        
OR = {

    "x1":[0,0,1,1],
    "x2":[0,1,0,1],
    "y":[0,1,1,1]
}

dataframes = pd,pd.DataFrame(OR)



if __name__ == "__main__":
    print(dataframes)