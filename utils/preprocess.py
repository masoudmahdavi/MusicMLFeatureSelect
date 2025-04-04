import pandas as pd
from model.model import Model
import logging
from tabulate import tabulate
from sklearn.model_selection import train_test_split

class Preprocess:
    def __init__(self, model:Model, logger:logging.Logger):
        self.model = model
        self.logger = logger
        self.logger.info("Preprocess class initialized.")
        self.raw_data = self.load_csv()
        self._describe_data()
    
    def _describe_data(self):
        """Print the description of the data."""
        null_counts = self.raw_data.isnull().sum().reset_index()
        null_counts.columns = ["Column Name", "Null Rows"]
        null_counts['type'] = null_counts['Column Name'].apply(lambda x: self.raw_data[x].dtype)
        self.logger.info('\n'+tabulate(null_counts, headers="keys", tablefmt="pretty"))
        
    def preprocess_raw_data(self):
        """Preprocess the raw data.
        This includes loading the data, handling missing values, normalizing data, and splitting into train/test sets.
        """
        self.logger.info("Starting data preprocessing...")
        features, target = self.split_featres_target()
        preprocessed_data = self.split_train_test(features, target)
        self.logger.info("Data preprocessing completed.")
        return preprocessed_data

    def split_featres_target(self):
        """Split the data into features and target variable.

        Returns:
            tuple: A tuple containing features and target variable.
        """
        features = self.raw_data.drop(columns=['Class'])
        target = self.raw_data['Class']
        self.logger.info("Data split into features and target variable.")
        return features, target
    
    def handle_miss_data(self):
        pass

    def normalize_data(self):
        pass

    def split_train_test(self, features:pd.DataFrame, target:pd.DataFrame) -> dict:
        def log_tabulate_data_shapes(dict_data):
            """Log the shapes of the dataframes in a tabular format."""
            df = pd.DataFrame()
            for key, value in dict_data.items():
                new_df = pd.DataFrame({'shape': [value.shape]}, index=[key])
                df = pd.concat([df, new_df], ignore_index=False)
            self.logger.info('\n'+tabulate(df, headers="keys", tablefmt="pretty"))
            del df, new_df

        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
        dict_data = {'X_train':X_train,
                     'X_test':X_test,
                     'y_train':y_train, 
                     'y_test':y_test}
        log_tabulate_data_shapes(dict_data)
        exit()
        self.logger.info("Data split into train and test sets.")
        self.logger.info(f"X_train shape: {X_train.shape}")
        self.logger.info(f"X_test shape: {X_test.shape}")
        self.logger.info(f"y_train shape: {y_train.shape}")
        self.logger.info(f"y_test shape: {y_test.shape}")
        self.logger.info('\n'+tabulate(X_train.head(), headers="keys", tablefmt="pretty"))
        self.logger.info('\n'+tabulate(X_test.head(), headers="keys", tablefmt="pretty"))
        self.logger.info('\n'+tabulate(y_train.head(), headers="keys", tablefmt="pretty"))  
        self.logger.info('\n'+tabulate(y_test.head(), headers="keys", tablefmt="pretty"))
        self.logger.info("Data split completed.")
        return dict_data

    def load_csv(self) -> pd.DataFrame:
        """Read data from csv file.

        Raises:
            ValueError: If data is empty

        Returns:
            pd.DataFrame: Dataframe containing data
        """
        data = pd.read_csv(self.model.data_path)
        if data.empty:
            raise ValueError("Data is empty")
        return data