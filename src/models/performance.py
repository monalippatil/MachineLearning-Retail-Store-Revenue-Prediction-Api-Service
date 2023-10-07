import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae

# Defining a function to determine the Mean Absolute Percentage Error (MAPE) performance score
def evaluating_mape_score(actual_revenues, predicted_revenues):
    """
    Calculating and returing the MAPE performance measure the percentage difference between the actual and predicted revenues of the forecasting model
    
    Parameters
    ----------
    actual_revenues : Numpy Array
        Actual target revenue values
    predicted_revenues : Numpy Array
        Predicted target revenue values

    Returns
    -------
    MAPE score : float
        Calculated Mean Absolute Percentage Error (MAPE) performance measure
    """  

    # Calculating the Mean Absolute Percentage Error (MAPE) performance measure of the forecasting model
    return round(np.mean(np.abs((actual_revenues - predicted_revenues)/actual_revenues)) * 100, 4)


# Defining a function to determine the Mean Absolute Error (MAE) performance score
def evaluating_mas_score(actual_revenues, predicted_revenues):
    """
    Calculating and returing the MAE performance measure the percentage difference between the actual and predicted values of the predictive model
    
    Parameters
    ----------
    actual_revenues : Numpy Array
        Actual target revenue values
    predicted_revenues : Numpy Array
        Predicted target revenue values

    Returns
    -------
    MAE score : float
        Calculated Mean Absolute Error (MAE) performance measure
    """  

    # Calculating the Mean Absolute Error (MAE) performance measure of the predictive model
    return round(mae(actual_revenues, predicted_revenues), 4)


# Defining a function to determine the Root Mean Square Error (RMSE) performance score
def evaluating_rmse_score(actual_revenues, predicted_revenues):
    """
    Calculating and returing the MAE performance measure the percentage difference between the actual and predicted values of the ML model
    
    Parameters
    ----------
    actual_revenues : Numpy Array
        Actual target revenue values
    predicted_revenues : Numpy Array
        Predicted target revenue values

    Returns
    -------
    RMSE score : float
        Calculated Root Mean Square Error (RMSE) performance measure
    """  

    # Calculating the Root Mean Square Error (RMSE) performance measure of the ML model
    return round(np.sqrt(mse(actual_revenues, predicted_revenues)), 4)