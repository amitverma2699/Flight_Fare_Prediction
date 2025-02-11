import os
import sys
import pandas as pd
import numpy as np
from Source.logger import logging
from Source.exception import CustomException
from Source.utils.utils import load_object

class FlightData:
    """
    Custom class to handle user input data for flight fare prediction.
    """

    def __init__(self, Airline, Source, Destination, Route, Dep_Time, Arrival_Time, Duration, Total_Stops, Additional_Info, Date_of_Journey):
        self.Airline = Airline
        self.Source = Source
        self.Destination = Destination
        self.Route = Route
        self.Dep_Time = Dep_Time
        self.Arrival_Time = Arrival_Time
        self.Duration = Duration
        self.Total_Stops = Total_Stops
        self.Additional_Info = Additional_Info
        self.Date_of_Journey = Date_of_Journey

    def get_data(self):
        """
        Returns input data as a dictionary.
        """
        try:
            flight_details = {
                "Airline": self.Airline,
                "Source": self.Source,
                "Destination": self.Destination,
                "Route": self.Route,
                "Dep_Time": self.Dep_Time,
                "Arrival_Time": self.Arrival_Time,
                "Duration": self.Duration,
                "Total_Stops": self.Total_Stops,
                "Additional_Info": self.Additional_Info,
                "Date_of_Journey": self.Date_of_Journey
            }
            return flight_details
        except Exception as e:
            logging.error("Error occurred while getting flight data.")
            raise CustomException(e, sys)

    def get_data_as_dataframe(self):
        """
        Converts input data into a Pandas DataFrame.
        """
        try:
            flight_details = self.get_data()
            df = pd.DataFrame([flight_details])
            return df
        except Exception as e:
            logging.error("Error occurred while converting data to DataFrame.")
            raise CustomException(e, sys)

class FlightFarePrediction:
    def __init__(self):
        try:
            self.model_path = os.path.join("Artifacts", "model.pkl")
            self.preprocessor_path = os.path.join("Artifacts", "preprocessor.pkl")

            logging.info("Loading saved model and preprocessor...")
            self.model = load_object(self.model_path)
            self.preprocessor = load_object(self.preprocessor_path)

        except Exception as e:
            logging.error("Error occurred while loading model or preprocessor.")
            raise CustomException(e, sys)

    def preprocess_input(self, input_data: pd.DataFrame):
        """
        Preprocess user input to match the model's training data.
        """
        try:
            df = input_data.copy()

            # Feature Engineering (same as in `DataTransformation`)
            df['Journey_Day'] = pd.to_datetime(df['Date_of_Journey'], format='%d/%m/%Y').dt.day
            df['Journey_Month'] = pd.to_datetime(df['Date_of_Journey'], format='%d/%m/%Y').dt.month
            df.drop('Date_of_Journey', axis=1, inplace=True)

            df['Dep_Hour'] = pd.to_datetime(df['Dep_Time']).dt.hour
            df['Dep_Minute'] = pd.to_datetime(df['Dep_Time']).dt.minute
            df.drop('Dep_Time', axis=1, inplace=True)

            df['Arrival_Hour'] = pd.to_datetime(df['Arrival_Time']).dt.hour
            df['Arrival_Minute'] = pd.to_datetime(df['Arrival_Time']).dt.minute
            df.drop('Arrival_Time', axis=1, inplace=True)

            df["Duration_Hour"] = df["Duration"].str.extract(r'(\d+)h')[0].astype(float)
            df["Duration_Minute"] = df["Duration"].str.extract(r'(\d+)m')[0].fillna(0).astype(float)
            df.drop(['Duration'], axis=1, inplace=True)

            logging.info("User input preprocessed successfully.")
            return df

        except Exception as e:
            logging.error("Error occurred while preprocessing user input.")
            raise CustomException(e, sys)

    def predict(self, input_data: pd.DataFrame):
        """
        Predict flight fare based on user input.
        """
        try:
            processed_data = self.preprocess_input(input_data)
            transformed_data = self.preprocessor.transform(processed_data)
            
            prediction = self.model.predict(transformed_data)
            predicted_price = round(prediction[0], 2)

            logging.info(f"Prediction Successful! Estimated Flight Price: {predicted_price}")
            return predicted_price

        except Exception as e:
            logging.error("Error occurred while making prediction.")
            raise CustomException(e, sys)
