import streamlit as st
from utils.movements import next_page,get_to_login
import os
import pickle
import pandas as pd
import os
from io import BytesIO
from pyxlsb import open_workbook as open_xlsb
import joblib
import glob



def app():
    st.markdown('<h2 style="color: #5b61f9;">Choose a model</h2>',
                unsafe_allow_html=True)

    w_dir = os.getcwd()



    dir_name = str(st.session_state.hash) + "_Models"



    #st.write(glob.glob(w_dir + "/"+ dir_name +"/*" + "/*")[1])




    features_directory = glob.glob(w_dir + "/"+ dir_name +"/*" + "/*")



    dict = {}

    if len(features_directory) > 0:

        if (features_directory[0][-3:]) == "txt":



            i = 1
            models = []
            #labels = []
            while i < len(features_directory):
                models.append([features_directory[i], i])

                dict[str(features_directory[i]).split("/")[-2]] = [features_directory[i], features_directory[i-1]]





                i +=2


            #models_name = []
            #for el in models:
                #models_name.
            a = st.selectbox("These are your saved models!", dict.keys())


            if a:
                st.markdown('<h2 style="color: #5b61f9;">Upload your Dataset and click Predict!</h2>',
                            unsafe_allow_html=True)



                if dict[a][1][-3:] == "txt":

                    with open(dict[a][1],errors="replace") as f:
                        mylist = list(f)
                else:

                    with open(dict[a][0],errors="replace") as f:
                        mylist = list(f)





        else:


            i = 0
            models = []
            #labels = []
            while i < len(features_directory):
                models.append([features_directory[i], i])
                dict[str(features_directory[i]).split("/")[-2]] = [features_directory[i], features_directory[i+1]]


                i +=2


            #models_name = []
            #for el in models:
                #models_name.
            a = st.selectbox("These are your saved models!", dict.keys())









            if a:
                st.markdown('<h2 style="color: #5b61f9;">Upload your Dataset and click Predict!</h2>',
                            unsafe_allow_html=True)

                if dict[a][1][-3:] == "txt":

                    with open(dict[a][1],errors="replace") as f:
                        mylist = list(f)
                else:

                    with open(dict[a][0],errors="replace") as f:
                        mylist = list(f)











                #st.write((mylist))

        if dict[a][0][-3:] =="sav":

            loaded_model = joblib.load(dict[a][0])
        else:
            loaded_model = joblib.load(dict[a][1])


        data_file = st.file_uploader("Only CSV file accepted", type=["csv","xls","xlsx"])

        try:

            if data_file != None:
                if data_file.name.split(".")[-1] == "csv":

                    # if check, warning
                    df = pd.read_csv(data_file)


                elif data_file.name.split(".")[-1] == "xlsx" or data_file.name.split(".")[-1] == "xls":
                    df = pd.read_excel(data_file)


                predict = st.button("Predict")
                if predict:
                    # if check, warning

                    ID = list(df["Customer ID"])

                    df = df.drop(["Customer ID","Target","Month"],axis=1)



                        #DATA PREPARATION
                    df1 = pd.DataFrame()
                    for el in mylist:
                        el = el[:-1]
                        if el != "Target" and el !="Month":
                            df1[el] = df[el]

                        #df = df[mylist]
                    categorical_var= df1.select_dtypes(exclude='number').columns.tolist()
                    numerical_var= df1.select_dtypes(include="number").columns.tolist()

                    if "Month" in categorical_var:
                            categorical_var.remove("Month")
                    if "Month" in numerical_var:
                            categorical_var.remove("Month")



                        # Top 9 classes + "other"
                    for el in df1[categorical_var].columns:
                        if len(list(df1[categorical_var][el].value_counts())) > 10:
                            a = list(df1[categorical_var][el].value_counts()[:10].index.tolist())
                            df1[el][df1[categorical_var][el].isin(a) == False] = "Other"




                    predictions = list((loaded_model.predict(df1)))

                    pred = pd.DataFrame()

                    pred["ID"] = ID
                    pred["predictions"] = predictions

                    st.write(pred)

                    def convert_df(df):
                        return df.to_csv().encode('utf-8')

                    csv = convert_df(pred)

                    st.download_button(
                        "Press to Download the Predictions!",
                        csv,
                        "predictions.csv",
                        "text/csv",
                        key='download-csv'
                    )

                    st.markdown('<h4 style="color: #5b61f9;">Columns Used</h4>',
                                unsafe_allow_html=True)

                    st.write(df1.columns)


            st.markdown('<h4 style="color: #ffffff;">Space</h4>',
                                unsafe_allow_html=True)


###
        except:
            st.warning("The dataset doesn't contain the right columns!")
            mylist = set(mylist) - set(["Target\n","Month\n"])

            mylist1 = []
            for el in mylist:
                mylist1.append(el[:-1])

            missing_columns = (set(mylist1) - (set(list(df.columns))))


            Columns_not_used = set(list(df.columns)) - set(mylist1)

            st.markdown('<h4 style="color: #5b61f9;">Missing columns</h4>',
                        unsafe_allow_html=True)

            st.write(missing_columns)

            st.markdown('<h4 style="color: #5b61f9;">Columns not recognized by the model</h4>',
                        unsafe_allow_html=True)

            st.write(Columns_not_used)




    else:

        st.warning("No Models trained found in your directory!")