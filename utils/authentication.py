import streamlit as st
from utils.movements import next_page
import os
import pandas as pd


#INITIALIZE SESSION STATE

if "light" not in st.session_state:
    st.session_state.light= "green"
if "warning" not in st.session_state:
    st.session_state.warning= "Undefined Error"
if "user" not in st.session_state:
    st.session_state.user= "User1"
if "hash" not in st.session_state:
    st.session_state.hash = 12345
if "new_user_succesfully_signed_in" not in st.session_state:
    st.session_state.new_user_succesfully_signed_in="no"
######################################################################


#DISCLAIMER: this code is saving the hash and the e-mail of the logged in users

def log_in_check(user,password):
    #check if a database named "users" exists, if not create one empty with just the hash column
    w_dir= os.getcwd()
    if "users.csv" not in os.listdir(w_dir):
        dic={
            "Hash": [12345],
            "E-mail": [email.prova @ test.com]
        }
        users=pd.DataFrame(data=dic)
        users.to_csv(w_dir+"/"+"users.csv")

    #if exist evaluate the hash of the user
    Hash = hash(str(user)+str(password))
    db = pd.read_csv(w_dir+"/"+"users.csv")["Hash"]

    # if it is in the db set the st.user with user and st.has as hash and set the light green
    if Hash in db.values:
        st.session_state.user=user
        st.session_state.hash= Hash
        st.session_state.light="green"

    # if is not in the db print an error message
    elif Hash not in db.values:
        st.session_state.warning="Invalid E-mail or Password"

    next_page()



def sign_in_user(user,password,password_2):
    #check if a database named "users" exists, if not create one empty with just the hash column
    w_dir= os.getcwd()
    if "users.csv" not in os.listdir(w_dir):
        dic={
            "Hash":[12345],
            "E-mail": ["email.prova@test.com"]
        }
        users=pd.DataFrame(data=dic)
        users.to_csv(w_dir+"/"+"users.csv")
    #check if the two passwords are the same
    if password==password_2:
        #check if a user is already signed in with the e-mail and print an already signed user error
        db=pd.read_csv(w_dir+"/"+"users.csv")
        mails= db["E-mail"]
        if user in mails.values:
            st.session_state.warning="This E-mail is already associated with a user"
            next_page()
        #if not create a new row and add it to the db
        elif user not in mails.values:
            new_user_data={
                "Hash":[hash(str(user)+str(password))],
                "E-mail":[str(user)]
            }
            new_user= pd.DataFrame(data=new_user_data)
            db=pd.concat([db,new_user],axis=0).drop(["Unnamed: 0"],axis=1)
            db.to_csv(w_dir+"/"+"users.csv")

            # Set light to green
            st.session_state.new_user_succesfully_signed_in="yes"

    elif password!=password_2:
        #print a different passwords error
        st.session_state.warning="Different Passwords!"
        next_page()
