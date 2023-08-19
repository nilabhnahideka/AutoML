import streamlit as st
import pandas as pd
import os
import ydata_profiling
from streamlit_pandas_profiling import st_profile_report
from pycaret.classification import setup, compare_models, pull, save_model, load_model, predict_model

def preprocess_data(data, feature_type):
    if feature_type == 'Categorical':
        return data.astype('category')
    else:
        return data.astype(float)

def main():
    st.sidebar.image("https://vitalflux.com/wp-content/uploads/2021/09/Python-automl-frameworks.png")
    st.sidebar.title("AutoStreamML")
    app_mode = st.sidebar.radio("Navigation", ["Main Tab"])
    st.sidebar.info("This application allows you to build an Automated Machine learning pipeline using Streamlit, Pandas Profiling, and PyCaret")

    if app_mode == "Main Tab":
        main_tab()

def main_tab():
    sub_tab = st.radio("Choose a Sub-Tab", ["Training", "Testing"])

    if sub_tab == "Training":
        training_sub_tab()
    elif sub_tab == "Testing":
        testing_sub_tab()

def training_sub_tab():
    st.title("Training Sub-Tab")
    
    with st.sidebar:
        st.title("AutoStreamML")
        choice = st.radio("Navigation",["Upload", "Profiling", "ML", "Download"])

    if os.path.exists("sourcedata.csv"):
        df = pd.read_csv("sourcedata.csv", index_col=None)

    if choice == "Upload":
        st.title("Upload your data for Modelling!")
        file = st.file_uploader("Upload your Dataset Here")
        if file:
            df = pd.read_csv(file, index_col=None)
            df.to_csv("sourcedata.csv", index=None)
            st.dataframe(df)

    if choice == "Profiling":
        st.title("Automated Exploratory Data Analysis")
        profile_report = df.profile_report()
        st_profile_report(profile_report)

    if choice == "ML":
        chosen_target = st.selectbox('Choose the Target Column', df.columns)
        feature_type = st.radio('Feature Data Type', ['Categorical', 'Numerical'])

        if st.button('Run Modeling'):
            setup_df = df.copy()

            if feature_type == 'Categorical':
                setup_df = setup_df.astype('category')
            else:
                setup_df = setup_df.astype(float)

            setup(setup_df, target=chosen_target, silent=True)
            st.dataframe(setup_df)
            best_model = compare_models()
            compare_df = pull()
            st.dataframe(compare_df)
            save_model(best_model, 'best_model')
            best_model

    if choice == "Download":
        with open('best_model.pkl', 'rb') as f:
            st.download_button('Download Model', f, file_name="best_model.pkl")

def testing_sub_tab():
    st.title("Testing Sub-Tab")
    
    model_filename = "best_model"
    trained_model = load_model(model_filename)

    prediction_data_file = st.file_uploader("Upload Data for Prediction", type="csv")
    if prediction_data_file:
        prediction_data = pd.read_csv(prediction_data_file)
        st.dataframe(prediction_data)

        feature_type = st.radio('Feature Data Type', ['Categorical', 'Numerical'])

        if st.button('Make Predictions'):
            preprocess_data(prediction_data, feature_type)
            predictions = predict_model(trained_model, data=prediction_data)
            st.dataframe(predictions)

if __name__ == "__main__":
    main()
