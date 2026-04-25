import pandas as pd
import logging
from sklearn.model_selection import train_test_split
from src.config import Config

logger = logging.getLogger(__name__)

def load_data(path: str) -> pd.DataFrame:
    """Load raw CSV data from the specified path."""
    try:
        return pd.read_csv(path)
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Clean data and create domain-specific features."""
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

def get_processed_split(data: pd.DataFrame):
    """Applies engineering and returns train/test splits."""
    df = feature_engineering(data)
    
    target = "Target" if "Target" in df.columns else df.columns[-1]
    X = df.drop(columns=[target])
    y = df[target]
    
    return train_test_split(X, y, test_size=Config.TEST_SIZE, random_state=Config.RANDOM_STATE)