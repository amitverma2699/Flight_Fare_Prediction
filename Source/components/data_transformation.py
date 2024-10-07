# data_transformation.py
import os
import sys
import pandas as pd
import numpy as np
from Source.logger import logging
from Source.exception import customexception
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from datetime import datetime
from Source.utils.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('Artifacts','preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
        pass

    def preprocess_data(self,df :pd.DataFrame):
        try:

            # 1. Handling missing values
            df.fillna(method='ffill', inplace=True)
        
            # 2. Feature engineering
            # Extracting the hour, minute, and day information from 'Date_of_Journey'
            df['Journey_Day'] = pd.to_datetime(df['Date_of_Journey']).dt.day
            df['Journey_Month'] = pd.to_datetime(df['Date_of_Journey']).dt.month
            df.drop('Date_of_Journey', axis=1, inplace=True)

            # Splitting 'Dep_Time' and 'Arrival_Time' into hours and minutes
            df['Dep_Hour'] = pd.to_datetime(df['Dep_Time']).dt.hour
            df['Dep_Minute'] = pd.to_datetime(df['Dep_Time']).dt.minute
            df.drop('Dep_Time', axis=1, inplace=True)

            df['Arrival_Hour'] = pd.to_datetime(df['Arrival_Time']).dt.hour
            df['Arrival_Minute'] = pd.to_datetime(df['Arrival_Time']).dt.minute
            df.drop('Arrival_Time', axis=1, inplace=True)

            # Converting 'Duration' into total minutes
            df['Duration_Minutes'] = df['Duration'].apply(self._convert_duration_to_minutes)
            df.drop('Duration', axis=1, inplace=True)

            # 3. Encoding categorical features
            df = self._encode_categorical_features(df)
            
            return df

            
        except Exception as e:
            logging.info("Exception Occured in the preprocess_data")
            raise customexception(e,sys)
    def _convert_duration_to_minutes(self, duration_str: str):
        try:
            duration = duration_str.split()
            total_minutes = 0
            for time in duration:
                if 'h' in time:
                    total_minutes += int(time.replace('h', '')) * 60
                elif 'm' in time:
                    total_minutes += int(time.replace('m', ''))
            return total_minutes
        except Exception as e:
            logging.info("Exception Occured in the Convertion of duration to minutes")
            raise customexception(e,sys)
        
        
    def _encode_categorical_features(self, df: pd.DataFrame):

        try:
            # One-hot encoding the categorical columns

            categorical_cols = ['Airline', 'Source', 'Destination', 'Route', 'Additional_Info']

            df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

            return df
        except Exception as e:
            logging.info("Exception Occured in the Categorical Features")
            raise customexception(e,sys)
    


    def scale_numerical_features(self, df: pd.DataFrame):
        try:

            numerical_cols = ['Journey_Day', 'Journey_Month', 'Dep_Hour', 'Dep_Minute',
                          'Arrival_Hour', 'Arrival_Minute', 'Duration_Minutes', 'Price']

            scaler = StandardScaler()
            df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

            return df
        except Exception as e:
            logging.info("Exception Occured in the Numerical Features")
            raise customexception(e,sys)