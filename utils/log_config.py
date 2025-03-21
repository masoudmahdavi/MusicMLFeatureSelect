import logging
import os
import sys
import argparse
from model.model import Model
from logging.handlers import RotatingFileHandler

def setup_logging(model:Model, args:argparse.Namespace) -> logging.Logger:
    """Set up logging configuration."""
    if args.rewrite:
            os.remove(model.log_file_dir + model.log_file_name)

    level = 'DEBUG' if args.verbos == True else 'INFO' 
    os.makedirs(model.log_file_dir, exist_ok=True)
    log_path = os.path.join(model.log_file_dir, model.log_file_name)

    # Configure logging
    logging.basicConfig(
        level=level,
        format = "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        handlers=[
            RotatingFileHandler(log_path, maxBytes=5_000_000, backupCount=3),  # 5MB log rotation
            logging.StreamHandler(sys.stdout)  # Log to console
        ]
    )

    return logging.getLogger("PipelineLogger")

    