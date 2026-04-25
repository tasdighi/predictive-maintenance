import pandas as pd
import logging
from sklearn.model_selection import train_test_split
from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(path: str) -> pd.DataFrame:
    try:
        data = pd.read_csv(path)
        logger.info(f"Data loaded successfully from {path}")
        return data
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # remove unnecessary columns
    cols_to_drop = ["Product ID", "UDI", "Failure Type"]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # replace [ and ] in column names with empty string
    df.columns = df.columns.str.replace('[\[\]]', '', regex=True) 
    
    # feature engineering
    if 'Process temperature K' in df.columns and 'Air temperature K' in df.columns:
        df['temp_diff'] = df['Process temperature K'] - df['Air temperature K']
    
    # convert categorical variables
    if "Type" in df.columns:
        df = pd.get_dummies(df, columns=["Type"], drop_first=True)
        
    return df

def get_train_test_data(data: pd.DataFrame):
    df = feature_engineering(data)
    
    target_col = "Target" if "Target" in df.columns else df.columns[-1]
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    return train_test_split(X, y, test_size=Config.TEST_SIZE, random_state=Config.RANDOM_STATE)