import streamlit as st
import os
import numpy as np
import pandas as pd
from MLProject.pipeline.prediction import PredictionPipeline


def main():
    # initialize prediction pipeline
    model = PredictionPipeline()

    # Set the title
    st.title("Loan Status Prediction")
    
    col1, col2 = st.columns([3,2])
    with col1:
        st.subheader("Enter the following details")
        # Get the input from the user
        st.number_input(label= "Current Loan Amount", 
                        key = "Current_Loan_Amount",
                        min_value= 10000,
                        max_value= 10000000,
                        step= 1000)

        st.selectbox(label= "Term",
                        key = "Term",
                        options = ['Short Term', 'Long Term'])

        st.number_input(label= "Credit Score",
                        key = "Credit_Score",
                        min_value= 585,
                        max_value= 7510,
                        step= 100)

        st.number_input(label= "Annual Income",
                        key = "Annual_Income",
                        min_value= 50000,
                        max_value= 9999999999,
                        step= 10000)

        st.selectbox(label= "Years in current job",
                        key = "Years_in_current_job",
                        options = ['8 years', '10+ years', '3 years', '5 years', 
                                '< 1 year', '2 years', '4 years', '9 years', 
                                '7 years', '1 year', '6 years'])

        st.selectbox(label= "Home Ownership",
                        key = "Home_Ownership",
                        options = ['Home Mortgage', 'Own Home', 'Rent', 'HaveMortgage'])

        st.selectbox(label= "Purpose",
                        key = "Purpose",
                        options = ['Home Improvements', 'Debt Consolidation', 'Buy House', 'other',
                                'Business Loan', 'Buy a Car', 'major_purchase', 'Take a Trip', 'Other',
                                'small_business', 'Medical Bills', 'wedding', 'vacation',
                                'Educational Expenses', 'moving', 'renewable_energy'])

        st.number_input(label= "Monthly Debt",
                        key = "Monthly_Debt",
                        min_value= 0,
                        max_value= 9999999999,
                        step= 1000)

        st.number_input(label= "Years of Credit History",
                        key = "Years_of_Credit_History",
                        min_value= 0,
                        max_value= 70,
                        step= 10)

        st.number_input(label= "Months since last delinquent",
                        key = "Months_since_last_delinquent",
                        min_value= 0,
                        max_value= 9999,
                        step= 10)

        st.number_input(label= "Number of Open Accounts",
                        key = "Number_of_Open_Accounts",
                        min_value= 0,
                        max_value= 999,
                        step= 10)

        st.number_input(label= "Number of Credit Problems",
                        key = "Number_of_Credit_Problems",
                        min_value= 0,
                        max_value= 999,
                        step= 10)

        st.number_input(label= "Current Credit Balance",
                        key = "Current_Credit_Balance",
                        min_value= 0,
                        max_value= 9999999999,
                        step= 10000)

        st.number_input(label= "Maximum Open Credit",
                        key = "Maximum_Open_Credit",
                        min_value= 0,
                        max_value= 9999999999,
                        step= 10000)

        st.number_input(label= "Bankruptcies",
                        key = "Bankruptcies",
                        min_value= 0,
                        max_value= 99,
                        step= 1)

        st.number_input(label= "Tax Liens",
                        key = "Tax_Liens",
                        min_value= 0,
                        max_value= 99,
                        step= 1)
        
    with col2:
        for i in range(5):
            st.write('\n')
        if st.button("Predict"):
            # Get the input from the user
            input_dict = {
                "Current_Loan_Amount" : st.session_state["Current_Loan_Amount"],
                "Term" : st.session_state["Term"],
                "Credit_Score" : st.session_state["Credit_Score"],
                "Annual_Income" : st.session_state["Annual_Income"],
                "Years_in_current_job" : st.session_state["Years_in_current_job"],
                "Home_Ownership" : st.session_state["Home_Ownership"],
                "Purpose" : st.session_state["Purpose"],
                "Monthly_Debt" : st.session_state["Monthly_Debt"],
                "Years_of_Credit_History" : st.session_state["Years_of_Credit_History"],
                "Months_since_last_delinquent" : st.session_state["Months_since_last_delinquent"],
                "Number_of_Open_Accounts" : st.session_state["Number_of_Open_Accounts"],
                "Number_of_Credit_Problems" : st.session_state["Number_of_Credit_Problems"],
                "Current_Credit_Balance" : st.session_state["Current_Credit_Balance"],
                "Maximum_Open_Credit" : st.session_state["Maximum_Open_Credit"],
                "Bankruptcies" : st.session_state["Bankruptcies"],
                "Tax_Liens" : st.session_state["Tax_Liens"]
            }
            st.write(f"The predicted result is {model.predict(input_dict)}")
            st.write(input_dict)
        st.info("Click on Predict button to get the predicted result")

    


if __name__ == "__main__":
    main()