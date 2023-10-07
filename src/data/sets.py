import pandas as pd
import numpy as np

# Defining a function to identify missing values in the dataset
def checking_null_values(df):
    """
    Identify if any null values in the dataset

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe

    Returns
    -------
    """
    
    # Checking the features if there are any null values
    print(df.isnull().sum())


# Defining a function to fill NaN values based on the feature's data type
def imputing_missing_values(df_data, attribute_name):
    """
    Impute the missing values of the attribute of the dataset

    Parameters
    ----------
    df_data : pd.DataFrame
        Input dataframe
    date_feature: object
        Name of the feature with the missing value
    Returns
    -------
    """
    
    # If the attribute is of object type (string), impute NaN with None
    if df_data[attribute_name].dtype == 'object':
        df_data[attribute_name].fillna('None', inplace=True)

    # For non-object types (numeric), impute NaN with 0
    else:
        df_data[attribute_name].fillna(0, inplace=True)


# Defining a function to extract date related information 
def extract_date_components(df_data, date_feature):
    """
    Deriving different date-related components from the date feature
    Parameters
    ----------
    df_data : pd.DataFrame
        Input dataframe
    date_feature: object
        Name of the date feature

    Returns
    -------
    """
        
    # Converting the date attribute to a datetime datatype
    df_data[date_feature] = pd.to_datetime(df_data[date_feature])
    
    # Extracting the day of the week, month, year, and week of the year information
    df_data['day_of_week'] = df_data[date_feature].dt.dayofweek
    df_data['month'] = df_data[date_feature].dt.month
    df_data['year'] = df_data[date_feature].dt.year
    df_data['week_of_year'] = df_data[date_feature].dt.isocalendar().week


# Defining a function to scale the numeric features to ensure uniformity in the feature's values
def features_scaling(df_dataset, numerical_features):
    """
    Scaling features to achieve consistency in feature values and returning the scaled dataset and scaler object
    
    Parameters
    ----------
    df_dataset : pd.DataFrame
        Input data dataframe
    numerical_features : pd.DataFrame
        Numerical features names dataframe

    Returns
    -------
    df_dataset : pd.DataFrame
        Scaled numeric features of the input dataset
    scaler : class 'sklearn.preprocessing._data.StandardScaler'
        Instantiated and fitted object of the StandardScaler
    """

    # Importing StandardScaler library to scale the features of the input dataset
    from sklearn.preprocessing import StandardScaler

    # Instantiating a instance of the StandardScaler named 'scaler'
    scaler = StandardScaler()
    
    # Fitting and applying the 'scaler' instance to perform Scaling the numerical features
    df_scaled = pd.DataFrame(scaler.fit_transform(df_dataset[numerical_features]), columns=df_dataset[numerical_features].columns)

    return df_scaled, scaler

# Defining a function to transform the categorical feature to numeric
def ordinal_transform(df_dataset, categorical_feature):
    """
    Transforming categorical feature to numeric to be utilised for training a Machine Learning model and returning the ordered feature and ordinal object
    
    Parameters
    ----------
  	df_dataset : pd.DataFrame
        Input data dataframe
    categorical_features : String
        Categorical feature name

    Returns
    -------
    df_ordered : pd.DataFrame
        Ordered categorical feature values to numeric
    ordinal : class 'sklearn.preprocessing._encoders.OrdinalEncoder'
        Instantiated and fitted object of the OrdinalEncoder
    """

    # Importing OrdinalEncoder library to transform categorical feature to numeric
    from sklearn.preprocessing import OrdinalEncoder

    # Instantiating a instance of the OrdinalEncoder named 'ordinal'
    ordinal = OrdinalEncoder()
    
    # Fitting and applying the 'ordinal' instance to convert categorical feature to numeric
    df_ordered = pd.DataFrame(ordinal.fit_transform(df_dataset[[categorical_feature]]), columns=[categorical_feature])

    return df_ordered, ordinal


# Defining a function to store all the processed datasets prepared for the machine learning purposes
def save_datasets(X_train=None, y_train=None, X_validate=None, y_validate=None, path='../../data/processed/'):
    """
    Store all the datasets locally in 'data/processed' directory that are prepared 

    Parameters
    ----------
    X_train: Numpy Array
        Features for the training dataset
    y_train: Numpy Array
        Target for the training dataset
    X_validate: Numpy Array
        Features for the validation dataset
    y_validate: Numpy Array
        Target for the validation dataset
    path : string
        Path to the folder where the sets will be saved (default: '../../data/processed/')

    Returns
    -------
    """

    # Saving the datasets individually
    if X_train is not None:
      np.save(f'{path}X_train', X_train)
    if X_validate is not None:
      np.save(f'{path}X_validate', X_validate)
    if y_train is not None:
      np.save(f'{path}y_train', y_train)
    if y_validate is not None:
      np.save(f'{path}y_validate', y_validate)