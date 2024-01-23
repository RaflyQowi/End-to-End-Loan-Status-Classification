import pandas as pd
import os
from MLProject import logger
from sklearn.svm import SVC
from MLProject.utils.common import save_bin
from MLProject.entity.config_entity import ModelTrainerConfig

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        logger.info(f"Loading training and test data.")
        train_df = pd.read_csv(self.config.train_data_path)
        test_df = pd.read_csv(self.config.test_data_path)

        logger.info(f"Splitting input and target column from training and test data")
        X_train = train_df.drop([self.config.target_column], axis= 1)
        X_test = test_df.drop([self.config.target_column], axis= 1)
        y_train = train_df[[self.config.target_column]].values.ravel()
        y_test = test_df[[self.config.target_column]]

        logger.info(f"Training Started")
        model = SVC(C = self.config.C,
                    kernel= self.config.kernel,
                    degree= self.config.degree,
                    random_state= 42)
        model.fit(X_train, y_train)

        save_bin(model, path = os.path.join(self.config.root_dir, self.config.model_name))