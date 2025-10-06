import logging
import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
def run_training_pipeline():
    """
    End-to-end training pipeline for KOI dataset.
    Steps:
    1. Data ingestion
    2. Data transformation
    3. Model training and evaluation
    4. Save best model and preprocessor
    Returns:
        dict: metrics of the best model (accuracy, f1, roc_auc)
    """
    try:
        logging.info("Starting training pipeline...")

        # Step 1: Data Ingestion
        data_ingestion = DataIngestion()
        train_path, test_path = data_ingestion.initiate_data_ingestion()
        logging.info("Data ingestion completed.")

        # Step 2: Data Transformation
        data_transformation = DataTransformation()
        train_array, test_array, preprocessor_path = data_transformation.initiate_data_transformation(
            train_path, test_path
        )
        logging.info("Data transformation completed.")

        # Step 3: Model Training
        model_trainer = ModelTrainer()
        accuracy, f1, roc_auc = model_trainer.initiate_model_trainer(train_array, test_array)
        logging.info(f"Model training completed. Metrics - Accuracy: {accuracy:.4f}, F1: {f1:.4f}, ROC-AUC: {roc_auc:.4f}")

        # Step 4: Return metrics and paths
        result = {
            "accuracy": accuracy,
            "f1_score": f1,
            "roc_auc": roc_auc,
            "preprocessor_path": preprocessor_path,
            "model_path": model_trainer.model_trainer_config.trained_model_file_path
        }

        logging.info("Training pipeline finished successfully.")
        return result

    except Exception as e:
        logging.error(f"Error occurred in training pipeline: {e}")
        raise e


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    metrics = run_training_pipeline()
    print(metrics)
    print(metrics['accuracy'], metrics['f1_score'], metrics['roc_auc'], metrics['preprocessor_path'], metrics['model_path'])