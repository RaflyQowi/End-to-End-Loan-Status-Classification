from MLProject.utils.common import load_bin,read_yaml
import numpy as np
import pandas as pd
from pathlib import Path
import os


class PredictionPipeline:
    def __init__(self):
        self.model = load_bin(Path('artifacts\model_trainer\model.joblib'))
        self.le = load_bin(Path('artifacts\data_transformation\label_encoders.bin'))
        self.sc = load_bin(Path('artifacts\data_transformation\scaler.bin'))
        self.col_info = read_yaml(Path("artifacts\data_transformation\column_info.yaml"))

    def predict(self, data:dict):
        data = pd.DataFrame([data])
        data = self.remove_unused_columns(data)
        data = self.get_inference_transfomation(data)
        prediction = self.model.predict(data)

        return self.le['Loan_Status'].inverse_transform(prediction)[0]
    
    def get_inference_transfomation(self, data):
        data[self.col_info['numeric_features']] = self.sc.transform(data[self.col_info['numeric_features']])
        for column in self.col_info['categorical_features']:
            try:
                data[column] = self.le[column].transform(data[column])
            except:
                pass
        return data
    
    def remove_unused_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        if os.path.exists("artifacts\data_transformation\column_info.yaml"):
            return data.drop(self.col_info.drop_feature_columns, axis= 1)
        
if __name__ == '__main__':
    pred = PredictionPipeline()
    data = {
  "Current_Loan_Amount": 10000,
  "Term": "Short Term",
  "Credit_Score": 585,
  "Annual_Income": 50000,
  "Years_in_current_job": "8 years",
  "Home_Ownership": "Home Mortgage",
  "Purpose": "Home Improvements",
  "Monthly_Debt": 0,
  "Years_of_Credit_History": 0,
  "Months_since_last_delinquent": 0,
  "Number_of_Open_Accounts": 0,
  "Number_of_Credit_Problems": 0,
  "Current_Credit_Balance": 0,
  "Maximum_Open_Credit": 0,
  "Bankruptcies": 0,
  "Tax_Liens": 0
}
    pred.predict(data)