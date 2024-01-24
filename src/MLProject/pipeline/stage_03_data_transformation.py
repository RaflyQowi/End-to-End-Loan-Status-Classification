from MLProject.config.configuration import ConfigurationManager
from MLProject.components.data_transformation import DataTransformation
from MLProject import logger
from pathlib import Path

STAGE_NAME = "Data Transformation Stage"

class DataTransformationTrainingPipeline:
    def __init__(self):
        config = ConfigurationManager()
        self.data_transformation_config = config.get_data_transformation_config()
        self.data_transformation = DataTransformation(config=self.data_transformation_config)

    def main(self):
        logger.info(f"{'>>'*20} {STAGE_NAME} Started {'<<'*20} \n\n")
        file_path = Path("artifacts/data_validation/status.txt")

        ## Check if validation is successful
        with open(file_path, 'r') as f:
            validation = [line.split()[-1] for line in f if line.strip()]
        final_status = 'False' if 'False' in validation else 'True'

        if final_status == 'True':
            self.data_transformation.clean_null_values()
            self.data_transformation.train_test_split()
            self.data_transformation.get_label_encoder()
            self.data_transformation.get_scaler()
            self.data_transformation.get_test_transormation()
            self.data_transformation.get_column_info()

            logger.info(f"{'>>'*20} {STAGE_NAME} Completed {'<<'*20} \n\n")
        
        else:
            logger.info(f"Data Validation Failed, Please check status.txt file for more details")
            raise Exception("Data Validation Failed, Please check status.txt file for more details")
    
if __name__ == '__main__':
    try:
        obj = DataTransformationTrainingPipeline()
        obj.main()
    except Exception as e:
        logger.exception(e)
        raise e
        