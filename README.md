# End-to-End Loan Status Classification

Welcome to the MLOps on Loan Status Classification Project! This repository serves as a comprehensive guide for implementing MLOps using scikit-learn libraries to train classification models. The project utilizes MLflow and DagsHub for experiment tracking and Docker Hub for deployment.

## Project Overview

In this project, we focus on the end-to-end workflow of MLOps, emphasizing the orchestration and deployment aspects rather than the model training itself.

## Project Setup

### 1. Update Configuration Files

Before getting started, customize the following essential configuration files:

- `config.yaml`: Modify global settings.
- `params.yaml`: Adjust parameters for model training and evaluation.
- `schema.yaml`: Utilize this for data validation, including column names and target variable types.

- `.env`: Update with your specific credentials and configurations:
  ```
  MLFLOW_TRACKING_URI=https://dagshub.com/RaflyQowi/End-to-End-Loan-Status-Classification.mlflow
  MLFLOW_TRACKING_USERNAME=your_username
  MLFLOW_TRACKING_PASSWORD=your_password
  ```

### 2. Update Entity and Configuration Manager

Ensure the entity and configuration manager, located in `src/config`, align with your project requirements.

### 3. Update Components

Review and customize various components to suit your specific use case.

### 4. Update Pipeline

Tailor the pipeline to match your dataset, model architecture, and training objectives.

### 5. Update Main Script

Customize the `main.py` script to incorporate any additional functionalities or specific requirements.

## How to Run

Follow these steps to set up and run the application:

### Step 1: Clone the Repository

```bash
git clone https://github.com/RaflyQowi/End-to-End-Loan-Status-Classification.git
```

### Step 2: Set Up Python Virtual Environment

```bash
python -m venv venv
source venv/bin/activate    # On Windows, use `.\venv\Scripts\activate`
```

### Step 3: Install Requirements

```bash
pip install -r requirements.txt
```

### Step 4: Run the Application

```bash
streamlit run app.py
```

### Step 5: Access Local Host

Open your local host and port in a web browser.

## How to Retrain the Model

### Step 1: Update Parameters

In the `params.yaml` file, update the parameters for the model.

### Step 2: Run Training

```bash
python main.py
```

### Step 3: View Experiments on MLflow

Set up your MLflow on DagsHub and then create an `.env` file.

For my experiments [here](https://dagshub.com/RaflyQowi/End-to-End-Loan-Status-Classification.mlflow).

Feel free to contribute or report issues. Happy coding!
