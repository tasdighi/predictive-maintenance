import os

class Config:
    DATA_PATH = "src/data/predictive_maintenance.csv"
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    MODEL_PARAMS = {
        'n_estimators': 100,
        'max_depth': None,
        'random_state': RANDOM_STATE
    }