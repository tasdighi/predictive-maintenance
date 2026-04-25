import os

class Config:
    # Paths
    DATA_PATH = "src/data/predictive_maintenance.csv"
    MODEL_SAVE_PATH = "models/best_model.joblib"
    
    # Model Hyperparameters
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    
    # Grid Search Space
    PARAM_GRID = {
        'clf__n_estimators': [50, 100, 200],
        'clf__max_depth': [None, 10, 20],
        'clf__min_samples_split': [2, 5]
    }