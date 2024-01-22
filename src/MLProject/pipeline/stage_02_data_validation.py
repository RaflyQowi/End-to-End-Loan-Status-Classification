from MLProject.config.configuration import ConfigurationManager
from MLProject.components.data_validation import DataValidation
from MLProject import logger

STAGE_NAME = "Data Validation Stage"

class DataValidationTrainingPipeline:
    def __init__(self):
        config = ConfigurationManager()
        self.data_validation_config = config.get_data_validation_config()
        self.data_validation = DataValidation(config=self.data_validation_config)

    def main(self):
        logger.info(f"{'>>'*20} {STAGE_NAME} Started {'<<'*20} \n\n")

        # Perform data validation
        valid_columns = self.data_validation.validate_all_columns()
        
        # If columns validation is successful, perform type validation; otherwise, skip type validation
        if valid_columns:
            valid_types = self.data_validation.validate_all_type()
        else:
            valid_types = self.data_validation.validate_all_type(initial=False)
        
        # Log data validation status
        validation_status = f"Data Validation status: columns validation is {valid_columns} and type validation is {valid_types}"
        logger.info(validation_status)
        logger.info(f"{'>>'*20} {STAGE_NAME} Completed {'<<'*20} \n\n")
    
if __name__ == '__main__':
    try:
        obj = DataValidationTrainingPipeline()
        obj.main()
    except Exception as e:
        logger.exception(e)
        raise e