import pandas as pd
from utils.preprocess import Preprocess
from model.model import Model
import mlflow
import pathlib
import logging
from utils.log_config import setup_logging


class SelectFeature:
    def __init__(self, model:Model):
        self.model = model
        self.preprocess = Preprocess(self.model)
        setup_logging(self.model)
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Data path: {self.model.data_path}")

    def preprocess_data(self) -> pd.DataFrame:
        preprocessed_data = self.preprocess.preprocess_raw_data()
        return preprocessed_data

if __name__ == "__main__":
    
    model = Model()
    model.data_path = 'data/Turkish_Music_Mood_Recognition.csv'
    model.log_file_dir = 'log/'
    model.log_file_name = 'pipeline.txt'
    select_feature_obj = SelectFeature(model)
    preprocessed_data = select_feature_obj.preprocess_data()