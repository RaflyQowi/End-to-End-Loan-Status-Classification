from MLProject import logger
import pandas as pd
from MLProject.entity.config_entity import DataValidationConfig

class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config
        self.data = pd.read_csv(self.config.unzip_data_dir)
        with open(self.config.STATUS_FILE, 'w') as f:
            pass
        
    def validate_all_columns(self) -> bool:
        try: 
            all_cols =self.data.columns.to_list()

            all_schema = self.config.all_schema.keys()

            validation_status = True  # Assume validation is True initially

            with open(self.config.STATUS_FILE, 'a') as f:
                for col in all_schema:
                    if col not in all_cols:
                        validation_status = False
                        f.write(f"Validation status: {validation_status}, Because {col} not in schema\n")
                
                # Write the final validation status to the file
                f.write(f"Final Validation Columns status: {validation_status}\n")

            return validation_status
            
        except Exception as e:
            logger.exception(e)
            raise e
    
    def validate_all_type(self, initial = True) -> bool:
        validation_status = initial  # Assume validation is True initially
        keys = self.config.all_schema.keys()
        all_cols = self.data.columns.to_list()
        with open(self.config.STATUS_FILE, 'a') as f:
            for col in all_cols:
                if self.config.all_schema[col] != self.data[col].dtype:
                    validation_status = False
                    f.write(f"Validation status: {validation_status}, Because {col} type different from schema\n")
        
            # Write the final validation status to the file
            f.write(f"Final Validation type status: {validation_status}\n")
        return validation_status