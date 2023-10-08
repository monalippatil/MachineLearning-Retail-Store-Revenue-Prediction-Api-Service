# Importing necessary libraries 
# 1. FastAPI package from fastapi required for the model's servicing
# 2. JSONResponse class required to convert dictionary responses to the JSON format
# 3. Load method to load the constructed models
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
def forecast_revenue(input_date='2015-01-05', days=6):
  
    # Loading the constructed predictive and forcasting model
    phophet_revenue = load('../models/forecasting/phophetrevenue_forecastingregressor.joblib')

    input_date = datetime.strptime(input_date, '%Y-%m-%d')
    future = input_date + dt.timedelta(days=days)

    dates = pd.date_range(start=input_date, end=future.strftime("%Y/%m/%d"),)
    df_date = pd.DataFrame({"ds": dates})

    forecast = phophet_revenue.predict(df_date)
    return_df = forecast[['ds','yhat']]

    projected_revenue = {}
    for data in return_df.to_dict("records"):
        date = data["ds"].strftime("%Y/%m/%d")
        projected_revenue[date] = data["yhat"]

    return projected_revenue

# Defining the '/' root endpoint and its response message 
@regressor_app.get("/")
def read_root():
    return JSONResponse(content=project_information)

# Defining the '/health' endpoint and its response message 
@regressor_app.get('/health/', status_code=200)
def healthcheck():
    return 'Welcome, to the Sales Prediction API!! The API is for Revenue Prediction and Forecasting Servicing.'

# Defining the '/sales/national/' endpoint and its response message
@regressor_app.get("/sales/national/")
def predict(date: str,): 
    ds = date
    result = forecast_revenue(ds)
    return result


@regressor_app.get("/sales/regressor/")
def predict_reg(item_id: str, store_id: str, event_name: str, event_type: str, indate: str,): 
    
    model_file = '../models/predictive/xgbrevenue_predictiveregressor.joblib'
    # if not model_file.exists():
    #     return False

    model = load(model_file)

    indate = datetime.strptime(indate, '%Y-%m-%d')
    day_of_week = indate.weekday()
    month = indate.month
    year = indate.year
    week_of_year = indate.isocalendar().week
    # outparam = create_df(item_id,store_id,event_name,event_type,indate, dayofweek, month, year, week_of_year)
    # outparam = list(item_id,store_id,event_name,event_type,indate, dayofweek, month, year, week_of_year)
    outparam = pd.DataFrame({'item_id':item_id,
                             'store_id': store_id,
                             'event_name': event_name,
                             'event_type': event_type,
                             'indate': indate, 
                             'day_of_week': day_of_week, 
                             'month': month, 
                             'year': year, 
                             'week_of_year': week_of_year}, index=[0])
    # newdf = pd.DataFrame(outparam.to_dict())
    result =  model.predict(outparam)
    # print(outparam)
    return result.tolist()