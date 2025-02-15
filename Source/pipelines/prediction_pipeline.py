import os
import sys
import pandas as pd
import numpy as np
from Source.logger import logging
from Source.exception import CustomException
from Source.utils.utils import load_object

class CustomData:
    """
    Custom class to handle user input data for flight fare prediction.
    """

    def __init__(self, Airline, Date_of_Journey, Source, Destination, 
                 Route, Dep_Time, Arrival_Time, 
                 Duration, Total_Stops, Additional_Info):
        
        self.Airline = Airline
        self.Date_of_Journey = Date_of_Journey
        self.Source = Source
        self.Destination = Destination
        self.Route = Route
        self.Dep_Time = Dep_Time
        self.Arrival_Time = Arrival_Time
        self.Duration = Duration
        self.Total_Stops = Total_Stops
        self.Additional_Info = Additional_Info
        
        
    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict={
                "Airline":[self.Airline],
                "Date_of_Journey":[self.Date_of_Journey],
                "Source":[self.Source],
                "Destination":[self.Destination],
                "Route":[self.Route],
                "Dep_Time":[self.Dep_Time],
                "Arrival_Time":[self.Arrival_Time],
                "Duration":[self.Duration],
                "Total_Stops":[self.Total_Stops],
                "Additional_Info":[self.Additional_Info]
                
            }

            df=pd.DataFrame(custom_data_input_dict)
            logging.info("DataFrame Created")
            return df

        except Exception as e:
            logging.info("Error occured in CustomData")
            raise CustomException(e,sys)
        
class PredictionPipeline:
    def __init__(self):
        try:
            preprocessor_path = os.path.join('artifacts','preprocessor.pkl')
            model_path = os.path.join('artifacts','model.pkl')
            
            self.preprocessor = load_object(preprocessor_path)
            self.model = load_object(model_path)
            
        except Exception as e:
            logging.error("Error occurred while loading preprocessor and model.")
            raise CustomException(e, sys)
        
    def preprocess_data(self, df: pd.DataFrame):
        try:
            logging.info("Starting data preprocessing.")

            # Extracting journey date features
            df['Journey_Day'] = pd.to_datetime(df['Date_of_Journey'], format='%d/%m/%Y').dt.day
            df['Journey_Month'] = pd.to_datetime(df['Date_of_Journey'], format='%d/%m/%Y').dt.month
            df.drop('Date_of_Journey', axis=1, inplace=True)

            # Extracting time features
            df['Dep_Hour'] = pd.to_datetime(df['Dep_Time']).dt.hour
            df['Dep_Minute'] = pd.to_datetime(df['Dep_Time']).dt.minute
            df.drop('Dep_Time', axis=1, inplace=True)

            df['Arrival_Hour'] = pd.to_datetime(df['Arrival_Time']).dt.hour
            df['Arrival_Minute'] = pd.to_datetime(df['Arrival_Time']).dt.minute
            df.drop('Arrival_Time', axis=1, inplace=True)

            # Processing 'Duration' feature
            df['Duration_Hour'] = df['Duration'].str.extract(r'(\d+)h').astype(float).fillna(0)
            df['Duration_Minute'] = df['Duration'].str.extract(r'(\d+)m').astype(float).fillna(0)
            df.drop(['Duration'], axis=1, inplace=True)

            logging.info("Data preprocessing completed.")
            return df
            
        except Exception as e:
            logging.error("Exception occurred in preprocess_data")
            raise CustomException(e, sys)
        
    def predict(self,features : pd.DataFrame):
        try:
            logging.info("Starting prediction.")
            processed_data = self.preprocess_data(features)
            transformed_data = self.preprocessor.transform(processed_data)
            prediction = self.model.predict(transformed_data)
            
            logging.info("Prediction completed.")
            
            return prediction
            
        except Exception as e:
            logging.error("Error occurred while predicting flight fare.")
            raise CustomException(e, sys)

        