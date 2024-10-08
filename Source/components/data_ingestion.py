import pandas as pd
import numpy as np
from Source.logger import logging
from Source.exception import customexception
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from pathlib import Path
import os
import sys


class DataIngestionConfig:
    train_data_path :str = os.path.join("Artifacts","train.csv")
    test_data_path :str = os.path.join("Artifacts","test.csv")

class DataIngestion():
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data ingestion started")

        try:
            train_data=pd.read_excel(Path(os.path.join("notebooks/data","train.xlsx")))
            test_data=pd.read_excel(Path(os.path.join("notebooks/data","test.xlsx")))

            logging.info("Load the train and test dataset")

            os.makedirs(os.path.dirname(os.path.join(self.ingestion_config.train_data_path)),exist_ok=True)
            train_data.to_csv(self.ingestion_config.train_data_path,index=False)
            os.makedirs(os.path.dirname(os.path.join(self.ingestion_config.test_data_path)),exist_ok=True)
            test_data.to_csv(self.ingestion_config.test_data_path,index=False)

            logging.info("Save the train and test data in Artifact folder")

            logging.info("Perform train test split")
            train_data,test_data=train_test_split(train_data,test_size=0.25,random_state=40)

            logging.info("train test split completed")

            train_data.to_csv(self.ingestion_config.train_data_path,index=False)
            test_data.to_csv(self.ingestion_config.test_data_path,index=False)

            logging.info("Data ingestion part completed")

            return (

                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )



        except Exception as e:
            logging.info("Exception during occured at Data Ingestion stage")
            raise customexception(e,sys)