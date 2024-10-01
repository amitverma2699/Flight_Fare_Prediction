from Source.components.data_ingestion import DataIngestion


import os
import sys
from Source.logger import logging
from Source.exception import customexception
import pandas as pd

obj=DataIngestion()

train_data_path,test_data_path=obj.initiate_data_ingestion()
