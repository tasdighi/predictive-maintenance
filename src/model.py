import mlflow
import mlflow.sklearn
import joblib
import logging
import os
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from src.config import Config

logger = logging.getLogger(__name__)

def get_pipeline() -> Pipeline:
    """Constructs the ML pipeline: Scaling -> Classification."""
    return Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(random_state=Config.RANDOM_STATE))
    ])

def train_and_optimize(X_train, y_train) -> Pipeline:
    """Performs Hyperparameter Tuning and logs results to MLflow."""
    pipeline = get_pipeline()
    
    with mlflow.start_run(run_name="Predictive_Maintenance_Training"):
        logger.info("Starting GridSearchCV...")
        
        grid_search = GridSearchCV(
            pipeline, 
            Config.PARAM_GRID, 
            cv=5, 
            scoring='f1', # F1 is often better for imbalanced maintenance data
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        
        # Log best parameters and score to MLflow
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("best_f1_score", grid_search.best_score_)
        
        # Log the actual model to MLflow artifact store
        mlflow.sklearn.log_model(grid_search.best_estimator_, "model")
        
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(Config.MODEL_SAVE_PATH), exist_ok=True)
 
        # Save locally for API use
        joblib.dump(grid_search.best_estimator_, Config.MODEL_SAVE_PATH)
        
        logger.info(f"Optimization complete. Best Params: {grid_search.best_params_}")
        return grid_search.best_estimator_

def evaluate(model: Pipeline, X_test, y_test):
    """Evaluates the model and logs metrics."""
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds)
    
    mlflow.log_metric("test_accuracy", acc)
    return acc, report