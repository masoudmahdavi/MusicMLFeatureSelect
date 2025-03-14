import pandas as pd
from model.model import Model

class Preprocess:
    def __init__(self, model:Model):
        self.model = model

    def preprocess_raw_data(self):
        raw_data = self.load_csv()
        return raw_data

    def handle_miss_data(self):
        pass

    def normalize_data(self):
        pass

    def split_data(self):
        pass

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