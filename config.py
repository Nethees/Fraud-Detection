


import os

# Configuration for path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'creditcard.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

os.makedirs(OUTPUT_DIR, exist_ok=True)

#General configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2


#Model Hyperparameters

LOGISTIC_REGRESSION_PARAMS = {
    'max_iter': 1000,
    'random_state': RANDOM_STATE,
    'class_weight': 'balanced'
}