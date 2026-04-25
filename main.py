import logging
from src.data.preprocess import load_data, get_processed_split
from src.model import train_and_optimize, evaluate
from src.config import Config

# Standard logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def run_pipeline():
    # 1. Ingestion
    raw_data = load_data(Config.DATA_PATH)
    
    # 2. Preprocessing & Splitting
    X_train, X_test, y_train, y_test = get_processed_split(raw_data)
    
    # 3. Training & Optimization (MLflow Tracking happens inside)
    best_model = train_and_optimize(X_train, y_train)
    
    # 4. Evaluation
    accuracy, report = evaluate(best_model, X_test, y_test)
    
    print("\n--- Final Results ---")
    print(f"Accuracy: {accuracy:.4f}")
    print("Report:\n", report)

if __name__ == "__main__":
    run_pipeline()