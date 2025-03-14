import pandas as pd
from utils.preprocess import Preprocess
from model.model import Model
import mlflow
import logging
from utils.log_config import setup_logging


class MLPipeline:
    def __init__(self, model:Model):
        self.model = model
        setup_logging(self.model) # Configure logging once
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Data path: {self.model.data_path}")

        self.preprocess = Preprocess(self.model)
    
    def run(self):
        """Execute the full ML pipeline."""
        self.logger.info("Starting the ML pipeline...")
        try:
            self.logger.info("Preprocessing data...")
            preprocessed_data = self.preprocess_data()
            
            self.logger.info("Data preprocessing completed.")
            self.logger.info("Logging preprocessed data to MLFlow...")
            # self.log_to_mlflow(preprocessed_data)
            self.logger.info("Data logged to MLFlow.")

        except Exception as e:
            self.logger.error(f"An error occurred: {e}")
        self.logger.info("ML pipeline completed.")

    def preprocess_data(self) -> pd.DataFrame:
        preprocessed_data = self.preprocess.preprocess_raw_data()
        return preprocessed_data

if __name__ == "__main__":
    
    model = Model()
    model.data_path = 'data/Turkish_Music_Mood_Recognition.csv'
    model.log_file_dir = 'log/'
    model.log_file_name = 'pipeline.txt'
    pipeline = MLPipeline(model)
    pipeline.run()
