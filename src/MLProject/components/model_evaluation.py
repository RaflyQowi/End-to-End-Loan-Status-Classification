import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
from MLProject.utils.common import load_bin, save_json
from MLProject.entity.config_entity import ModelEvaluationConfig
from pathlib import Path

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def log_into_mlflow(self):
        test_data = pd.read_csv(self.config.test_data_path)
        model = load_bin(self.config.model_path)

        X_test = test_data.drop([self.config.target_column], axis= 1)
        y_test = test_data[[self.config.target_column]]
        
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():

            predicted = model.predict(X_test)

            (accuracy, f1, precision, recall) = self._eval_metrics(y_test, predicted)

            ## Saving metrics as ocal
            scores = {
                "accuracy": accuracy,
                "f1": f1,
                "precision": precision,
                "recall": recall
            }
            save_json(path = Path(self.config.metric_file_path), data = scores)

            ## Log params
            mlflow.log_params(self.config.all_params)

            ## Log metrics
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("f1", f1)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)

            # Model registry does not work with file store
            if tracking_url_type_store != "file":
                # Register the model
                # There are other ways to use the Model Registry, which depends on the use case.
                # docs: https://mlflow.org/docs/latest/model-registry.html#api-workflow
                mlflow.sklearn.log_model(model, "model", registered_model_name="SVC")
            else:
                mlflow.sklearn.log_model(model, "model")

    @staticmethod
    def _eval_metrics(actual, pred):
        accuracy = accuracy_score(actual, pred)
        f1 = f1_score(actual, pred)
        precision = precision_score(actual, pred)
        recall = recall_score(actual, pred)
        return (accuracy, f1, precision, recall)