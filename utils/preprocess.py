import pandas as pd
from model.model import Model

class Preprocess:
    def __init__(self, model:Model):
        self.model = model

    def preprocess_raw_data(self):
        raw_data = self.read_csv()

    def handle_miss_data(self):
        pass

    def normalize_data(self):
        pass

    def split_data(self):
        pass

    def read_csv(self):
        data = pd.read_csv(self.model.data_path)
 