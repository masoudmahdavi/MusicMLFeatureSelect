import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.preprocess import Preprocess
from model.model import Model
import unittest


class PreprocessTest(unittest.TestCase):

    def setUp(self):
        self.model = Model()
        self.model.data_path = 'data/Turkish_Music_Mood_Recognition.csv'
        self.preprocess = Preprocess(self.model)
    
    def test_load_csv(self):
        data = self.preprocess.load_csv()
        self.assertIsNotNone(data)

    def test_preprocess_raw_data(self):
        data = self.preprocess.preprocess_raw_data()
        self.assertIsNotNone(data)

    # def test_handle_miss_data(self):
    #     self.preprocess.handle_miss_data()

    # def test_normalize_data(self):
    #     self.preprocess.normalize_data()

    # def test_split_data(self):
    #     self.preprocess.split_data()
