import streamlit as st
import pandas as pd

#SESSION STATE INITIALIZATION
if 'df' not in st.session_state:
    st.session_state.df = 0
if "light" not in st.session_state:
    st.session_state.light = "red"
if "warning" not in st.session_state:
    st.session_state.warning = "Undefined Error"
####################################################


def app():
    #Set the stop
    st.session_state.light = "red"
    st.session_state.warning= "Please upload a Dataset!"

    st.markdown(
        """
        <style>
            .stProgress > div > div > div > div {
                background-image: linear-gradient(to right, #5b61f9 , #5b61f9);
            }
        </style>""",
        unsafe_allow_html=True,
    )
    my_bar = st.progress(25)

    #App Body
    st.markdown('<h1 style="color: #5b61f9;">Home</h1>',
                unsafe_allow_html=True)

    # 3ab0e0
    #st.write('This is the `home page` of the app.')

    st.write('In this app, we will be building a simple classification model for your dataset.')

    st.markdown('<h3 style="color: #5b61f9;">Upload your Dataset and click Next!</h3>',
                unsafe_allow_html=True)
    data_file = st.file_uploader("Only CSV file accepted", type=["csv"])



    if data_file != None:
        #if check, warning
        df = pd.read_csv(data_file)

        list_columns = ["Target","Customer ID","Month"]
        columns = df.columns
        counter = 0
        for el in list_columns:
            if el not in columns or type(df["Month"]) == int:
                counter +=1

        if counter != 0:
            st.session_state.warning = "Dataset Import Error. The dataset should contain Target, Month and Customer ID column."
        else:
             #ADD CHECK FOR COLUMNS HERE
            st.session_state.df = df
            st.session_state.light= "green"
        st.dataframe(df)
