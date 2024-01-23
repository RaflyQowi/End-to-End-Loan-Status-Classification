from MLProject import logger
from MLProject.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from MLProject.pipeline.stage_02_data_validation import DataValidationTrainingPipeline
from MLProject.pipeline.stage_03_data_transformation import DataTransformationTrainingPipeline
from MLProject.pipeline.stage_04_model_trainer import ModelTrainerTrainingPipeline

try:
    obj = DataIngestionTrainingPipeline()
    obj.main()
except Exception as e:
    logger.exception(e)
    raise e

try:
    obj = DataValidationTrainingPipeline()
    obj.main()
except Exception as e:
    logger.exception(e)
    raise e

try:
    obj = DataTransformationTrainingPipeline()
    obj.main()
except Exception as e:
    logger.exception(e)
    raise e

try:
    obj = ModelTrainerTrainingPipeline()
    obj.main()
except Exception as e:
    logger.exception(e)
    raise e
    