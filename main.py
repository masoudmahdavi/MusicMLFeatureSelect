import pandas as pd
from utils.preprocess import Preprocess
from model.model import Model
import mlflow
import logging
from utils.log_config import setup_logging
import argparse

class MLPipeline:
    def __init__(self, model:Model, args:argparse.Namespace):
        
        self.model = model
        setup_logging(self.model, args) # Configure logging once
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Data path: {self.model.data_path}")
        self.preprocess = Preprocess(self.model, self.logger)
    
    def run(self):
        """Execute the full ML pipeline."""
        self.logger.info("Starting the ML pipeline...")
        # with mlflow.start_run():
        try:
            self.logger.info("Preprocessing data...")
            preprocessed_data = self.preprocess.preprocess_raw_data()
            # mlflow.log_artifact(self.model.data_path)
            self.logger.info("Data preprocessing completed.")
            self.logger.info("Logging preprocessed data to MLFlow...")
            # self.log_to_mlflow(preprocessed_data)
            self.logger.info("Data logged to MLFlow.")

        except Exception as e:
            if args.verbos:
                self.logger.error(f"An error occurred: {e}", exc_info=True)
                raise e
            else:
                self.logger.error(f"An error occurred: {e}")
        finally:
            self.logger.info("ML pipeline completed.")
            pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ML Pipeline')
    parser.add_argument('--verbos', '-v',action='store_true', help='Enable verbose logging')
    parser.add_argument('--rewrite', type=bool, help='Rewrite log file', default=False)
    args = parser.parse_args()

    model = Model()
    model.data_path = 'data/Turkish_Music_Mood_Recognition.csv'
    model.log_file_dir = 'log/'
    model.log_file_name = 'pipeline.txt'
    pipeline = MLPipeline(model, args)
    pipeline.run()


