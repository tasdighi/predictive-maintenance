import logging
from src.data.preprocess import load_data, get_train_test_data
from src.model import get_model_pipeline, train_model, evaluate_model
from config import Config

# تنظیمات لاگینگ برای کل پروژه
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    # 1. load data
    df = load_data(Config.DATA_PATH)
    
    # 2. data splitting and engineering
    X_train, X_test, y_train, y_test = get_train_test_data(df)
    
    # 3. create and train pipeline
    pipeline = get_model_pipeline()
    pipeline = train_model(pipeline, X_train, y_train)
    
    # 4. evaluate
    acc, report = evaluate_model(pipeline, X_test, y_test)
    
    print("\n" + "="*30)
    print(f"Final Model Accuracy: {acc:.4f}")
    print("Full Classification Report:")
    print(report)
    print("="*30)

if __name__ == "__main__":
    main()