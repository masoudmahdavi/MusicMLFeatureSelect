import pandas as pd
from utils.preprocess import Preprocess
from model.model import Model
import mlflow
import pathlib


class SelectFeature:
    def __init__(self, data_path:str):
        self.model = Model()
        self.preprocess = Preprocess(self.model)
        self.model.data_path = data_path
    
    def preprocess_data(self) -> pd.DataFrame:
        preprocessed_data = self.preprocess.preprocess_raw_data()
        return preprocessed_data

if __name__ == "__main__":
    data_path = 'data/Turkish_Music_Mood_Recognition.csv'
    select_feature_obj = SelectFeature(data_path)
    preprocessed_data = select_feature_obj.preprocess_data()