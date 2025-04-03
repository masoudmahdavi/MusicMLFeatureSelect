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
        pass

    def handle_miss_data(self):
        pass

    def normalize_data(self):
        pass

    def split_data(self):
        X_train, X_test, y_train, y_test = train_test_split(self.raw_data, test_size=0.2, random_state=42)
        

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