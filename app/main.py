# Importing necessary libraries 
# 1. FastAPI package from fastapi required for the model's servicing
# 2. JSONResponse class required to convert dictionary responses to the JSON format
# 3. Load method to load the constructed models
# 4 to 6: Pandas and datetime required for servicing model API
from fastapi import FastAPI
from starlette.responses import JSONResponse
from joblib import load
import pandas as pd
from datetime import datetime
import datetime as dt

# Creating a instance of the FastAPI
regressor_app = FastAPI()

# Define the project information dictionary
project_information = {"description": "This API provides predictive and forecasting models for sales revenue.", 
                        "endpoints": {"/": "GET - Display project information (this endpoint)", 
                                    "/health/": "GET - Check API health status", 
                                    "/sales/national/": "GET - Get sales volume forecast for the next 7 days", 
                                    "/sales/stores/items/": "GET - Get predicted sales volume for a specific item and store",},
                        "input_parameters": {"/sales/national/": {"date": "string (YYYY-MM-DD) - Date for which to forecast sales"},
                        "/sales/stores/items/": {"item_id": "string - ID of the item", 
                                         "store_id": "string - ID of the store", 
                                         "date": "string (YYYY-MM-DD) - Date for which to predict sales"}},
                        "output_format": {"/sales/national/": "JSON with forecasted sales volume", 
                                  "/sales/stores/items/": "JSON with predicted sales volume for the item and store"},
                        "github_repo": "https://github.com/MonaliPatil19/adv_mla_assignment2.git"
                        }

# Defining the function for utilization of the predictive model
def forecast_revenue(input_date='2015-01-05', days=6):
  
    # Loading the constructed forcasting model
    phophet_revenue = load('../models/forecasting/phophetrevenue_forecastingregressor.joblib')

    # Transforming the input date to the datetime datatype
    input_date = datetime.strptime(input_date, '%Y-%m-%d')
    future = input_date + dt.timedelta(days=days)

    # Preparing dataframe with input date named as 'ds' feature to be provided to the forecasting phophet model 
    dates = pd.date_range(start=input_date, end=future.strftime("%Y/%m/%d"),)
    df_date = pd.DataFrame({"ds": dates})

    # Predicting the sales revenue employing forcasting item
    forecast = phophet_revenue.predict(df_date)

    # Extracting only dates and the corresponding sales revenue values from the 'yhat' attribute
    return_df = forecast[['ds','yhat']]

    # Building a dataframe to hold 7 days dates and its respective forecasted sales income
    projected_revenue = {}
    for data in return_df.to_dict("records"):
        date = data["ds"].strftime("%Y/%m/%d")
        projected_revenue[date] = data["yhat"]

    # Displaying the forecasted sales revenue for 7 days to the user
    return projected_revenue

# Defining the '/' root endpoint and its response message 
@regressor_app.get("/")
def read_root():
    return JSONResponse(content=project_information)

# Defining the '/health' endpoint and its response message 
@regressor_app.get('/health/', status_code=200)
def healthcheck():
    return 'Welcome, to the Sales Prediction API!! The API is for Revenue Prediction and Forecasting Servicing.'

# Defining the '/sales/national/' endpoint and its response message - forecasting related endpoint
@regressor_app.get("/sales/national/")
def predict(date: str,): 
    ds = date
    result = forecast_revenue(ds)
    return result

# Defining the '/sales/stores/items/' endpoint and its response message - prediction related endpoint
@regressor_app.get("/sales/stores/items/")
def predict_reg(item_id: str, store_id: str, event_name: str, event_type: str, date: str,): 
    
    # Loading the constructed predictive model
    xgboost_revenue = load('../models/predictive/xgbrevenue_predictiveregressor.joblib')
    
    # Deriving date related information from the data input field
    # Extracting the day of the week, month, year, and week of the year information
    indate = datetime.strptime(date, '%Y-%m-%d')
    day_of_week = indate.weekday()
    month = indate.month
    year = indate.year
    week_of_year = indate.isocalendar().week

    # Preparing the dataframe with the features to be provided to the predictive xgboost model
    model_parameters = pd.DataFrame({'item_id':item_id,
                             'store_id': store_id,
                             'event_name': event_name,
                             'event_type': event_type,
                             'indate': indate, 
                             'day_of_week': day_of_week, 
                             'month': month, 
                             'year': year, 
                             'week_of_year': week_of_year}, index=[0])
    
    # Predicting the sales revenue employing predictive item
    result =  xgboost_revenue.predict(model_parameters)
    
    # Displaying the predicted sales revenue to the user
    return result.tolist()