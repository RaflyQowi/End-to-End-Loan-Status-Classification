from MLProject.config.configuration import ConfigurationManager
from MLProject.components.data_ingestion import DataIngestion
from MLProject import logger

STAGE_NAME = "Data Ingestion Stage"

class DataIngestionTrainingPipeline:
    def __init__(self):
        config = ConfigurationManager()
        self.data_ingestion_config = config.get_data_ingestion_config()
        self.data_ingestion = DataIngestion(config=self.data_ingestion_config)

    def main(self):
        logger.info(f"{'>>'*20} {STAGE_NAME} Started {'<<'*20} \n\n")
        self.data_ingestion.download_data()
        self.data_ingestion.extract_zip_file()
        logger.info(f"{'>>'*20} {STAGE_NAME} Completed {'<<'*20} \n\n")

if __name__ == '__main__':
    try:
        obj = DataIngestionTrainingPipeline()
        obj.main()
    except Exception as e:
        logger.exception(e)
        raise e