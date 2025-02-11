import os
import sys
import pandas as pd
from Source.logger import logging
from Source.exception import CustomException
from Source.utils.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            preprocessor_path=os.path.join("Artifacts","preprocessor.pkl")
            model_path=os.path.join("Artifacts","model.pkl")

            preprocessor=load_object(preprocessor_path)
            model=load_object(model_path)

            scaled_data=preprocessor.transform(features)
            prediction=model.predict(scaled_data)

            return prediction

        except Exception as e:
            logging.info("Error occured in Prediction Pipeline")
            raise CustomException(e,sys)
        

class CustomData:
    def __init__(self,
                 carat:float,
                 depth:float,
                 table:float,
                 x:float,
                 y:float,
                 z:float,
                 cut:str,
                 color:str,
                 clarity:str,
                 ):
        self.carat=carat
        self.depth=depth
        self.table=table
        self.x=x
        self.y=y
        self.z=z
        self.cut=cut
        self.color=color
        self.clarity=clarity

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict={
                "carat":[self.carat],
                "depth":[self.depth],
                "table":[self.table],
                "x":[self.x],
                "y":[self.y],
                "z":[self.z],
                "cut":[self.cut],
                "color":[self.color],
                "clarity":[self.clarity]
            }

            df=pd.DataFrame(custom_data_input_dict)
            logging.info("DataFrame Created")
            return df

        except Exception as e:
            logging.info("Error occured in CustomData")
            raise CustomException(e,sys)
        
                 
                 
                 
                 
                 