from Source.pipelines.prediction_pipeline import CustomData, PredictionPipeline
import os
import pandas as pd

data=CustomData("SpiceJet","24/04/2019","New Delhi","Cochin","DEL → MAA → COK","15:45","22:05","6h 20m","1 stop","No info")

print(data.get_data_as_dataframe())