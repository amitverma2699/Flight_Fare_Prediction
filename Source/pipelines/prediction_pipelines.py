import os
import sys
import pandas as pd
import numpy as np
from Source.logger import logging
from Source.exception import CustomException
from Source.utils.utils import load_object


class PredictionPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            preprocessor_path = os.path.join('artifacts','preprocessor.pkl')
            model_path = os.path.join('artifacts','model.pkl')
            
            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)
            
            scaled_data=preprocessor.transform(features)
            prediction = model.predict(scaled_data)
            
            return prediction
            
        except Exception as e:
            logging.error("Error occurred while predicting flight fare.")
            raise CustomException(e, sys)
        
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
        
        
