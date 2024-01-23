from MLProject.config.configuration import ConfigurationManager
from MLProject.components.model_evaluation import ModelEvaluation
from MLProject import logger

STAGE_NAME = "Model Evaluation Stage"

class ModelEvaluationTrainingPipeline:
    def __init__(self):
        config = ConfigurationManager()
        self.model_evaluation_config = config.get_model_evaluation_config()
        self.model_evaluation = ModelEvaluation(config=self.model_evaluation_config)

    def main(self):
        logger.info(f"********** {STAGE_NAME} started **********")
        self.model_evaluation.log_into_mlflow()
        logger.info(f"********** {STAGE_NAME} completed **********")

if __name__ == "__main__":
    try:
        obj = ModelEvaluationTrainingPipeline()
        obj.main()
    except Exception as e:
        logger.exception(e)
        raise e