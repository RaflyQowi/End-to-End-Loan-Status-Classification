from MLProject.config.configuration import ConfigurationManager
from MLProject.components.model_trainer import ModelTrainer
from MLProject import logger


STAGE_NAME = "Model Training Stage"

class ModelTrainerTrainingPipeline:
    def __init__(self):
        config = ConfigurationManager()
        self.model_trainer_config = config.get_model_trainer_config()
        self.model_trainer = ModelTrainer(config=self.model_trainer_config)
    
    def main(self):
        logger.info(f"{'>>'*20} {STAGE_NAME} Started {'<<'*20} \n\n")
        self.model_trainer.train()
        logger.info(f"{'>>'*20} {STAGE_NAME} Completed {'<<'*20} \n\n")

if __name__ == '__main__':
    try:
        obj = ModelTrainerTrainingPipeline()
        obj.main()
    except Exception as e:
        logger.exception(e)
        raise e