from MLProject import logger
from MLProject.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from MLProject.pipeline.stage_02_data_validation import DataValidationTrainingPipeline

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