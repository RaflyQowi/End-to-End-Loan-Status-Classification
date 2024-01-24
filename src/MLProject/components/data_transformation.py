import pandas as pd
from sklearn.impute import SimpleImputer
from scipy.stats import shapiro
import warnings
import os
from pathlib import Path
from MLProject import logger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler
from MLProject.utils.common import save_bin, load_bin
from MLProject.entity.config_entity import DataTransformationConfig
import yaml

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        data = pd.read_csv(self.config.data_path)
        self.data = self._rename_columns(data)
        self.categorical_features, self.numeric_features, self.identifier = self._get_features_type(self.data)
        self.data = self.data.drop(self.identifier, axis = 1)
        self.train_path = os.path.join(self.config.root_dir, "train.csv")
        self.test_path = os.path.join(self.config.root_dir, "test.csv")
        self.sc_path = os.path.join(self.config.root_dir, "scaler.bin")
        self.le_path = os.path.join(self.config.root_dir, "label_encoders.bin")
        self.drop_feature_columns = None

    def get_column_info(self):
        logger.info(f"Saving column info at: {os.path.join(self.config.root_dir, 'column_info.yaml')}")
        data_dict = {
            "categorical_features": self.categorical_features,
            "numeric_features": self.numeric_features,
            "identifier": self.identifier,
            "drop_feature_columns": self.drop_feature_columns
        }

        with open(os.path.join(self.config.root_dir, "column_info.yaml"), 'w') as f:
            yaml.dump(data_dict, f)
    
    def get_test_transormation(self):
        data = pd.read_csv(self.test_path)
        le = load_bin(self.le_path)
        sc = load_bin(self.sc_path)

        logger.info(f"Starting transforming test data")
        data[self.numeric_features] = sc.transform(data[self.numeric_features])
        for column in le.keys():
            data[column] = le[column].transform(data[column])
        
        logger.info(f"Transforming test data completed")
        self._save_df(data, self.test_path)

    def get_label_encoder(self):
        logger.info(f"Label encoding started")
        data =pd.read_csv(self.train_path)
        label_encoders = {}
        for column in self.categorical_features:
            le = LabelEncoder()
            data[column] = le.fit_transform(data[column])
            label_encoders[column] = le
        
        self._save_df(data, self.train_path)
        save_bin(label_encoders, self.le_path)
        logger.info(f"Label encoding completed")

    def get_scaler(self):
        logger.info(f"Scaling numerical features started")
        data =pd.read_csv(self.train_path)
        scaler = RobustScaler()
        data[self.numeric_features] = scaler.fit_transform(data[self.numeric_features])

        self._save_df(data, self.train_path)
        save_bin(scaler, self.sc_path)
        logger.info(f"Scaling numerical features completed")

    def train_test_split(self):
        # Split the data into training and test sets. (0.75, 0.25) split.
        train, test = train_test_split(self.data, test_size=0.25, random_state=42, shuffle=True)

        logger.info(f"Save splitted data to {self.config.root_dir}")
        self._save_df(train, self.train_path)
        self._save_df(test, self.test_path)
        logger.info(f"Save CSV files Completed")

        logger.info("Split data into training and test sets")
        logger.info(f"train data shape: {train.shape}")
        logger.info(f"test data shape: {test.shape}")
        
    
    def clean_null_values(self):
        logger.info(f"Null values cleaning started")
        logger.info(f"Null values before cleaning: {self.data.isnull().sum().sum()}| Shape: {self.data.shape}")
        self.drop_feature_columns, drop_null_value_columns, imputation_columns = self._get_missing_values_strategies(self.data)
        self.data.drop(self.drop_feature_columns, axis=1, inplace=True)
        self.data.dropna(subset=drop_null_value_columns, inplace=True)
        self.numeric_features = [x for x in self.numeric_features if x not in self.drop_feature_columns]
        self.categorical_features = [x for x in self.categorical_features if x not in self.drop_feature_columns]
        self.data = self._impute_missing_values(self.data, imputation_columns, self.categorical_features)
        logger.info(f"Null values cleaned")
        logger.info(f"Null values after cleaning: {self.data.isnull().sum().sum()}| Shape: {self.data.shape}")
    
    @staticmethod
    def _save_df(data, path):
        data.to_csv(path, index=False)
        logger.info(f"Saving data at: {Path(path)}")

    
    def _impute_missing_values(self, df, columns: list, categorical_features: list):
        for column in columns:
            if column in categorical_features:
                imputer = SimpleImputer(strategy= 'most_frequent')
                df[columns] = imputer.fit_transform(df[columns])
                logger.info(f"{column} imputed with most frequent value")
            else:
                df[column] = self._impute_based_on_distribution(df[column])
        return df
    
    @staticmethod
    def _impute_based_on_distribution(data, imputation_strategy='mean', alpha=0.05):
        # Suppress the warning for the Shapiro-Wilk test
        warnings.filterwarnings("ignore", category=UserWarning)

        # Determine if the distribution is normal
        stat, p_value = shapiro(data.dropna())
        is_normal = p_value > alpha

        # Choose imputation strategy based on distribution
        if is_normal:
            imputer = SimpleImputer(strategy=imputation_strategy)
            imputed_data = imputer.fit_transform(data)
            logger.info(f"{data.name} imputed with {imputation_strategy}")
        else:
            imputed_data = data.fillna(data.median())
            logger.info(f"{data.name} imputed with median")

        return imputed_data
    
    @staticmethod
    def _get_missing_values_strategies(df, drop_feature_threshold=50, imputation_threshold=10):
        # Count missing values
        missing_counts = df.isnull().sum()

        # Percentage of missing values
        missing_percentage = (missing_counts / len(df) * 100).astype('float')

        # Identify columns for each strategy
        drop_feature_columns = missing_counts[missing_percentage >= drop_feature_threshold].index.tolist()
        imputation_columns = missing_counts[(missing_percentage < drop_feature_threshold) & (missing_percentage >= imputation_threshold)].index.tolist()
        drop_null_value_columns = missing_counts[missing_percentage < imputation_threshold].index.tolist()

        return drop_feature_columns, drop_null_value_columns, imputation_columns

    @staticmethod
    def _get_features_type(data):
        identifier = ['Loan_ID', 'Customer_ID']
        columns_use = data.drop(identifier, axis = 1).columns
        categorical_features = [feature for feature in columns_use if data[feature].dtype == 'O']
        numeric_features = [feature for feature in columns_use if feature not in categorical_features]
        return categorical_features, numeric_features, identifier

    @staticmethod
    def _rename_columns(data):
        col = data.columns.str.replace(' ', '_')
        data.columns = col
        return data