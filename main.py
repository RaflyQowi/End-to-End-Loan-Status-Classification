from MLProject import logger
from MLProject.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline

try:
    obj = DataIngestionTrainingPipeline()
    obj.main()
except Exception as e:
    logger.exception(e)
    raise e