import os
import sys
import pandas as pd
import numpy as np
from Source.logger import logging
from Source.exception import CustomException
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from Source.utils.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('Artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        
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

    
    def get_data_transformation(self):
        try:
            logging.info("Starting data transformation pipeline.")

            # Defining categorical and numerical columns
            categorical_cols = ['Airline', 'Source', 'Destination', 'Route', 'Additional_Info']
            ordinal_col = ['Total_Stops']
            ordinal_categories = [['non-stop', '1 stop', '2 stops', '3 stops', '4 stops']]

            numerical_cols = ['Journey_Day', 'Journey_Month', 'Dep_Hour', 'Dep_Minute',
                              'Arrival_Hour', 'Arrival_Minute', 'Duration_Hour', 'Duration_Minute']

            # Creating pipelines
            cat_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(handle_unknown='ignore',sparse_output=False)),
                ('scaler', StandardScaler(with_mean=False))
            ])

            ordinal_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OrdinalEncoder(categories=ordinal_categories))
            ])

            num_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ])

            # Column transformer to combine pipelines
            preprocessor = ColumnTransformer([
                ('num_pipeline', num_pipeline, numerical_cols),
                ('cat_pipeline', cat_pipeline, categorical_cols),
                ('ordinal_pipeline', ordinal_pipeline, ordinal_col)
            ])

            logging.info("Data transformation pipeline created successfully.")
            return preprocessor
        
        except Exception as e:
            logging.error("Exception occurred in get_data_transformation")
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            logging.info("Starting data transformation process.")

            # Load train and test datasets
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Train and test data loaded successfully.")
            logging.info(f"Train dataframe head: \n{train_df.head().to_string()}")
            logging.info(f"Test dataframe head: \n{test_df.head().to_string()}")

            # Apply preprocessing
            train_df = self.preprocess_data(train_df)
            test_df = self.preprocess_data(test_df)

            # Get transformation object
            preprocessing_obj = self.get_data_transformation()

            # Splitting features and target
            target_column_name = 'Price'
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]
            
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            # Transform data
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            logging.info("Applied preprocessing object on training and testing datasets.")

            # Combine transformed features with target
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            # Save preprocessing object
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            logging.info("Preprocessing pickle file saved successfully.")

            return train_arr, test_arr

        except Exception as e:
            logging.error("Exception occurred in initiate_data_transformation")
            raise CustomException(e, sys)
