import streamlit as st
import pandas as pd
from utils.df_description import check
import os
from datetime import datetime

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
    st.markdown('<h1 style="color: #5b61f9;">Select a Dataframe</h1>',
                unsafe_allow_html=True)

    # 3ab0e0
    #st.write('This is the `home page` of the app.')

    st.write('In this app, we will be building a simple classification model for your dataset.')

    st.markdown('<h3 style="color: #5b61f9;">Upload your Dataset and click Next!</h3>',
                unsafe_allow_html=True)
    data_file = st.file_uploader("Only CSV and Excel file accepted", type=["csv","xls","xlsx"])

    #data_xls = pd.read_excel('your_workbook.xls', 'Sheet1', index_col=None)
    #data_xls.to_csv('your_csv.csv', encoding='utf-8')

    if data_file != None:
        if data_file.name.split(".")[-1] == "csv":

        #if check, warning
            df = pd.read_csv(data_file)

        elif data_file.name.split(".")[-1] == "xlsx" or data_file.name.split(".")[-1] == "xls":
            df = pd.read_excel(data_file)

        list_columns = ["Target","Customer ID","Month"]
        columns = df.columns
        counter = 0
        for el in list_columns:
            if el not in columns:
                counter +=1

        if counter != 0:
            st.session_state.warning = "Dataset Import Error. The dataset should contain Target, Month and Customer ID column."
        else:
             #ADD CHECK FOR COLUMNS HERE
            st.session_state.df = df
            st.session_state.light= "green"
        st.dataframe(df)

        w_dir = os.getcwd()
        dir_name="Descriptions"
        if dir_name not in os.listdir(w_dir):
            os.mkdir(w_dir + "/" + dir_name)

        time = str(datetime.now().strftime("%H.%M.%S %d-%m-%Y"))
        save_path = w_dir + "/" + dir_name + "/" + time + "_" +"Df_description.xlsx"


        if st.button("Press to Download a Statistic Description of your Dataset!"):
            check(df,save_path)


        # describe = df.describe()
        #
        #
        # def convert_df(df):
        #     return df.to_csv().encode('utf-8')
        #
        #
        # describe_csv = convert_df(describe)
        #
        # st.markdown('<h3 style="color: #ffffff;">Upload your Dataset and click Next!</h3>',
        #             unsafe_allow_html=True)
        #
        #
        # st.download_button(
        #     "Press to Download a Statistic Description of your Dataset!",
        #     describe_csv,
        #     "Descriptive.csv",
        #     "text/csv",
        #     key='download-csv'
        # )
        #
        # st.markdown('<h3 style="color: #ffffff;">Upload your Dataset and click Next!</h3>',
        #             unsafe_allow_html=True)

