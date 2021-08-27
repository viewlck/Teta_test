import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.write("""
# Uplift Prediction App
This app predicts the uplift for clients!
Data obtained from the [X5 Retail Hero: Uplift Modeling for Promotional Campaign](https://ods.ai/competitions/x5-retailhero-uplift-modeling/data).
""")

st.markdown("""
[Example CSV input file](https://raw.githubusercontent.com/viewlck/Teta_test/main/example.csv)
""")

# Collects user input features into dataframe
uploaded_file = st.file_uploader("Choose a file", type=['csv'])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file,index_col='client_id')
    st.write(input_df)
    ind = input_df.index
    model = joblib.load('model.pkl')
    uplift = model.predict(input_df)
    df = pd.DataFrame(data={'uplift':uplift},index = ind)
    st.write(df)
