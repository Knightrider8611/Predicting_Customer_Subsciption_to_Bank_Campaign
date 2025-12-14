import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import xgboost

# 1. ตั้งค่า Model และ Data
st.set_page_config(page_title="Bank Deposit Prediction", layout="wide")
st.title("Bank Deposit Prediction App (One-Hot Support)")

@st.cache_resource
def load_model():

    path = r'.\Models\model_without_feature_en.pkl'
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as file:
        model = pickle.load(file)
    return model


model = load_model()

if model is None:
    st.error("ไม่พบไฟล์ Model กรุณาเช็ค Path ในโค้ด")
    st.stop()


MODEL_COLUMNS = [
    'age', 'default', 'balance', 'housing', 'loan', 'campaign', 'pdays', 'previous', 
    'month_num', 
    'categorical__job_admin.', 'categorical__job_blue-collar', 'categorical__job_entrepreneur', 
    'categorical__job_housemaid', 'categorical__job_management', 'categorical__job_retired', 
    'categorical__job_self-employed', 'categorical__job_services', 'categorical__job_student', 
    'categorical__job_technician', 'categorical__job_unemployed', 'categorical__job_unknown',
    'categorical__marital_divorced', 'categorical__marital_married', 'categorical__marital_single',
    'categorical__education_primary', 'categorical__education_secondary', 'categorical__education_tertiary', 'categorical__education_unknown',
    'categorical__contact_cellular', 'categorical__contact_telephone', 'categorical__contact_unknown',
    'categorical__month_apr', 'categorical__month_aug', 'categorical__month_dec', 'categorical__month_feb',
    'categorical__month_jan', 'categorical__month_jul', 'categorical__month_jun', 'categorical__month_mar', 
    'categorical__month_may', 'categorical__month_nov', 'categorical__month_oct', 'categorical__month_sep',
    'categorical__poutcome_failure', 'categorical__poutcome_other', 'categorical__poutcome_success', 'categorical__poutcome_unknown'
]


st.sidebar.header("Fill in customer information.")


age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=30)
balance = st.sidebar.number_input("Balance", value=0)
campaign = st.sidebar.number_input("Campaign (จำนวนครั้งที่ติดต่อ)", min_value=1, value=1)
pdays = st.sidebar.number_input("Pdays (วันหลังติดต่อครั้งก่อน)", value=-1)
previous = st.sidebar.number_input("Previous (จำนวนครั้งที่เคยติดต่อ)", value=0)


job = st.sidebar.selectbox("Job", [
    'admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 'retired',
    'self-employed', 'services', 'student', 'technician', 'unemployed', 'unknown'
])

marital = st.sidebar.selectbox("Marital Status", ['married', 'single', 'divorced'])
education = st.sidebar.selectbox("Education", ['primary', 'secondary', 'tertiary', 'unknown'])
default = st.sidebar.selectbox("Credit Default?", ['no', 'yes'])
housing = st.sidebar.selectbox("Housing Loan?", ['yes', 'no'])
loan = st.sidebar.selectbox("Personal Loan?", ['no', 'yes'])
contact = st.sidebar.selectbox("Contact Type", ['cellular', 'telephone', 'unknown'])
month = st.sidebar.selectbox("Month", [
    'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'
])
poutcome = st.sidebar.selectbox("Poutcome", ['unknown', 'failure', 'other', 'success'])



if st.button("Predict"):
    
    input_data = {col: 0 for col in MODEL_COLUMNS}
    
    
    input_data['age'] = age
    input_data['balance'] = balance
    input_data['campaign'] = campaign
    input_data['pdays'] = pdays
    input_data['previous'] = previous
    
    
    input_data['default'] = 1 if default == 'yes' else 0
    input_data['housing'] = 1 if housing == 'yes' else 0
    input_data['loan'] = 1 if loan == 'yes' else 0
    
    
    months_map = {
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
        'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
    }
    input_data['month_num'] = months_map[month]
    
    
    def set_one_hot(prefix, value):
        col_name = f"{prefix}_{value}"
        if col_name in input_data:
            input_data[col_name] = 1
            
    set_one_hot('categorical__job', job)
    set_one_hot('categorical__marital', marital)
    set_one_hot('categorical__education', education)
    set_one_hot('categorical__contact', contact)
    set_one_hot('categorical__month', month)
    set_one_hot('categorical__poutcome', poutcome)


 
    input_df = pd.DataFrame([input_data])
    
    
    input_df = input_df[MODEL_COLUMNS]

    try:
        prediction = model.predict(input_df)
        
        try:
            proba = model.predict_proba(input_df)
            confidence = np.max(proba) * 100
        except:
            confidence = 0

        
        st.divider()
        result = prediction[0]
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Result:")
            if result == 1:
                st.success("Yes")
            else:
                st.error("No")
                
        with col2:
            st.metric("Confidence", f"{confidence:.2f}%")
            
        with st.expander(" Model (Debug)"):
            st.dataframe(input_df)

    except Exception as e:
        st.error(f"Error {e}")
        if hasattr(model, 'n_features_in_'):
             st.warning(f"Model need {model.n_features_in_} but we {input_df.shape[1]} features.")


st.divider()
st.header(" Model Insights")


try:
    
    feature_names = MODEL_COLUMNS 
    importances = model.feature_importances_
    
    
    feat_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    
   
    feat_df = feat_df.sort_values(by='Importance', ascending=False).head(10)
    
    
    st.bar_chart(feat_df.set_index('Feature'))
    st.caption("Graph showing the top 10 factors that most influence the model's decision.")

except Exception as e:
    st.error(f"Can not plot graph: {e}")