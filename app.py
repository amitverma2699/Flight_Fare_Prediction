from Source.pipelines.prediction_pipelines import CustomData

custom_data_obj = CustomData('SpiceJet','24/04/2019','New Delhi','Cochin','DEL → MAA → COK','15:45','22:05','6h 20m','1 stop','No info')

print(custom_data_obj.get_data_as_dataframe())