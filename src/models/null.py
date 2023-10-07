import numpy as np
import pandas as pd

# Defining a class to access the baseline performance
class NullAccuracy:
    """
    Class used as baseline model for the regression task

    Attributes
    ----------
    y : Numpy Array-like
        Target variable
    y_mean : Float
        Value to be used for prediction
    y_base : Numpy Array
        Predicted array

    Methods
    -------
    fit(y)
        Store the input target variable and calculate the predicted value to be used
    predict(y)
        Generate the predictions
    fit_predict(y)
        Perform a fit followed by predict
    """
    # Method to declare the self attributes of the class
    def __init__(self):
        self.y = None
        self.y_mean = None
        self.y_base = None

    # Method to calculate the mean of the target variables
    def fit(self, y):
        self.y = y
        self.y_mean = y.mean()

    # Method to create a array consisting of mean value
    def predict(self, y):
        self.y_base = np.full((len(y), 1), self.y_mean)
        return self.y_base

    # Method to predict the target variable classes
    def fit_predict(self, y):
        self.fit(y)
        return self.predict(self.y)