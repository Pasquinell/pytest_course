import pandas as pd
from sklearn.model_selection import train_test_split
from .config import TRAIN_DATA, TEST_DATA
import logging


def load_data():
    train_df = pd.read_csv(TRAIN_DATA)
    test_df = pd.read_csv(TEST_DATA)
    return train_df, test_df
