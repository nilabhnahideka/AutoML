import streamlit as st
import pandas as pd
import os
import ydata_profiling
from streamlit_pandas_profiling import st_profile_report

from pycaret.regression import setup, compare_models, pull, save_model


with st.sidebar:
    st.image("https://vitalflux.com/wp-content/uploads/2021/09/Python-automl-frameworks.png")
    st.title("AutoStreamML")
    choice = st.radio("Navigation",["Upload", "Profiling", "ML", "Download"])
    st.info("This application allows you to build an Automated Machine learning pipleine using Streamlit, Pandas Profiling and PyCaret")

if os.path.exists("sourcedata.csv"):
    df = pd.read_csv("sourcedata.csv", index_col= None)

if choice == "Upload":
    st.title("Upload your data for Modelling!")
    file = st.file_uploader("Upload your Dataset Here")
    if file:
        df = pd.read_csv(file, index_col= None)
        df.to_csv("sourcedata.csv", index= None)
        st.dataframe(df)


if choice == "Profiling":
    st.title("Automated Exploratory Data Analysis")
    profile_report = df.profile_report()
    st_profile_report(profile_report)


if choice == "ML":
    chosen_target = st.selectbox('Choose the Target Column', df.columns)
    if st.button('Run Modelling'): 
        setup_df = df.copy()
        setup_df[chosen_target] = setup_df[chosen_target].astype(float)
        setup(setup_df, target=chosen_target, silent=True)
        setup_df[chosen_target] = setup_df[chosen_target].astype('category')
        st.dataframe(setup_df)
        best_model = compare_models()
        compare_df = pull()
        st.dataframe(compare_df)
        save_model(best_model, 'best_model')
        best_model

if choice == "Download":
   with open('best_model.pkl', 'rb') as f: 
        st.download_button('Download Model', f, file_name="best_model.pkl")
