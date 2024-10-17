import os
import sys
import pandas as pd
import numpy as np
from Source.logger import logging
from Source.exception import customexception
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
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

    def get_data_transformation(self,df: pd.DataFrame):
        try:
            logging.info("Data Transformation Initiated")
            df.fillna(method='ffill', inplace=True)
        
            #Feature engineering
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
            
            # OneHotEncoding  features
            ohe=OneHotEncoder()
            features_array=ohe.fit_transform(df[["Airline","Source","Destination"]]).toarray()
            feature_labels=ohe.categories_
            feature_labels=np.array(feature_labels,dtype=object).ravel()
            feature_labels=np.concatenate(feature_labels)
            features=pd.DataFrame(features_array,columns=feature_labels)
            df=pd.concat([df,features],axis=1)
            df.drop(["Airline", "Source", "Destination"], axis = 1, inplace = True)
            
            Total_stops=['non-stop','1 stop','2 stops','3 stops','4 stops']
            categorical_cols=[  'Airline_Air India', 'Airline_GoAir', 'Airline_IndiGo',
                                'Airline_Jet Airways', 'Airline_Jet Airways Business',
                                'Airline_Multiple carriers',
                                'Airline_Multiple carriers Premium economy', 'Airline_SpiceJet',
                                'Airline_Trujet', 'Airline_Vistara', 'Airline_Vistara Premium economy',
                                'Source_Chennai', 'Source_Kolkata', 'Source_Mumbai', 'Source_New Delhi',
                                'Destination_Cochin', 'Destination_Hyderabad', 'Destination_Kolkata',
                                'Destination_New Delhi']

            cat_pipeline=Pipeline(
                 steps=[
                      ('imputer',SimpleImputer(strategy='most_frequent')),
                      ('OneHotEncoder',OneHotEncoder())
                      ('ordinalencoding',OrdinalEncoder(categories=[Total_stops])),
                      ('scaler',StandardScaler())
                 ]
            )


            # Encoding numerical features
            numerical_cols = ['Journey_Day', 'Journey_Month', 'Dep_Hour', 'Dep_Minute',
                          'Arrival_Hour', 'Arrival_Minute', 'Duration_Minutes', 'Price']
            num_pipeline=Pipeline(
                 steps=[
                      ('imputer',SimpleImputer(strategy='medium')),
                      ('scaler',StandardScaler())
                 ]
            )
            

            preprocessor=ColumnTransformer([
                 
                 ('Num_pipeline',num_pipeline,numerical_cols),
                 ('Cat_pipeline',cat_pipeline,categorical_cols)
            ])

            return preprocessor
        
        except Exception as e:
            logging.info("Exception Occured in the get_data_transformation")
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
                

    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("read train and test data complete")
            logging.info(f"Train dataframe head : \n{train_df.head().to_string()}")
            logging.info(f"Test dataframe head : \n{test_df.head().to_string()}")

            preprocessing_obj = self.get_data_transformation()

            target_column_name='Price'
            drop_columns = [target_column_name,"Route","Additional_info"]

            input_feature_train_df = train_df.drop(columns=drop_columns,axis=1)
            target_feature_train_df=train_df[target_column_name]
            
            
            input_feature_test_df=test_df.drop(columns=drop_columns,axis=1)
            target_feature_test_df=test_df[target_column_name]
            
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)
            
            logging.info("Applying preprocessing object on training and testing datasets.")
            
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            
            logging.info("preprocessing pickle file saved")
            
            return (
                train_arr,
                test_arr
            )

        except Exception as e:
            logging.info("Exception occured in the initiate_data_transformation")
            raise customexception(e,sys)
    