import os
import streamlit as st
import pickle
from datetime import datetime
import shutil
import base64
import os.path
import joblib

#Initialize session state ---------------> This variables will be used to name the model
if "model" not in st.session_state:
    st.session_state.model= "To_be_selected"
if "user" not in st.session_state:
    st.session_state.user= "User1"
if "hash" not in st.session_state:
    st.session_state.hash = 12345
############################################################


def save_model(model):
    df = st.session_state.df[st.session_state.selected_var]
    columns = df.columns
    #get current directory and understand if the folder has already been created
    w_dir= os.getcwd()
    dir_name= str(st.session_state.hash)+"_Models"

    # if not create it with name: hash_models
    if dir_name not in os.listdir(w_dir):
        os.mkdir(w_dir+"/"+dir_name)




    #if yes create the path to save the new model as datetime_modelname_username
    time = str(datetime.now().strftime("%H.%M %d-%m-%Y"))

    os.mkdir(w_dir + "/" + dir_name + "/" + st.session_state.model + "_" + time)



    save_path = w_dir + "/" + dir_name + "/" + st.session_state.model +"_" + time

    model_path = save_path + "/" +st.session_state.user + ".sav"


    #create the pickle file and save it there
    joblib.dump(model, model_path)


    name_of_file = ("Z" + st.session_state.model + "_" + st.session_state.user + "Features" +".txt")


    save_path1 = save_path + "/" + name_of_file



    file1 = open(save_path1, "w")


    df = st.session_state.df[st.session_state.selected_var]

    for el in df.columns:
        file1.write(el +"\n")


    #st.write(str(df.columns))







def create_download_button_models():
    w_dir= os.getcwd()
    dir_name= str(st.session_state.hash) +"_Models"
    #1
    zip_directory= w_dir+"/"+dir_name
    #2
    zip_destination= w_dir+"/"+"compressed_models"
    #3
    filename= st.session_state.user +"_Models.zip"
    #Check if user has some model to export
    if not os.path.exists(zip_directory):
        pass
    elif os.path.exists(zip_directory):
        #Make zip folder and tranform it in bytes
        shutil.make_archive(zip_destination, 'zip', zip_directory)
        with open(zip_destination+".zip", 'rb') as f:
            bytes = f.read()
        #Download button
        st.download_button("Download all",bytes,filename)

        #Erease the zip folder to not occupy space
        if os.path.exists(zip_destination+".zip"):
            os.remove(zip_destination+".zip")