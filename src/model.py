import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from config import Config

logger = logging.getLogger(__name__)

def get_model_pipeline() -> Pipeline:
    """create a machine learning pipeline with scaling and random forest classifier"""
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(**Config.MODEL_PARAMS))
    ])
    return pipeline

def train_model(pipeline: Pipeline, X_train, y_train) -> Pipeline:
    logger.info("Starting model training...")
    pipeline.fit(X_train, y_train)
    logger.info("Training completed.")
    return pipeline

def evaluate_model(pipeline: Pipeline, X_test, y_test):
    predictions = pipeline.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)
    return acc, report