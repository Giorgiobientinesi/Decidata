from sqlalchemy import create_engine
import pandas as pd
from sqlalchemy import insert

from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select

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
    st.session_state.hash = 0
if "new_user_succesfully_signed_in" not in st.session_state:
    st.session_state.new_user_succesfully_signed_in="no"


def log_in_check(user,password):
    def myHash(text:str):
        hash = 0
        for ch in text:
            hash = (hash * 281 ^ ord(ch) * 997) & 0xFFFFFFFF
        return hash


    engine = create_engine(
        "postgresql://nwfwwqvcmctyqy:2ee23e2aa78021bedaa6e16d66d741e22043b5841d1724792d9f36f77f4bb47e@ec2-34-248-169-69.eu-west-1.compute.amazonaws.com:5432/d431jc0pi2nn2r",
        echo=False)

    Hash = myHash(str(user)+str(password))
    Hashes = []

    result_set = engine.execute("SELECT * FROM Users")
    for el in result_set:
        Hashes.append(str(el[0]))




    if str(Hash) in Hashes:
        st.session_state.user=user
        st.session_state.hash= password
        st.session_state.light="green"

    # if it is in the db set the st.user with user and st.has as hash and set the light green

    # if is not in the db print an error message
    elif str(Hash) not in Hashes:
        st.session_state.warning="Invalid E-mail or Password"

    next_page()



def sign_in_user(user,password,password_2):

    def myHash(text:str):
        hash = 0
        for ch in text:
            hash = (hash * 281 ^ ord(ch) * 997) & 0xFFFFFFFF
        return hash

    engine = create_engine(
        "postgresql://nwfwwqvcmctyqy:2ee23e2aa78021bedaa6e16d66d741e22043b5841d1724792d9f36f77f4bb47e@ec2-34-248-169-69.eu-west-1.compute.amazonaws.com:5432/d431jc0pi2nn2r",
        echo=False)

    #check if a database named "users" exists, if not create one empty with just the hash column
    #check if the two passwords are the same
    if password==password_2:
        Hash1 = myHash(str(user) + str(password))
        Hashes1 = []


        result_set = engine.execute("SELECT * FROM Users")
        for el in result_set:
            Hashes1.append(str(el[0]))


        if str(Hash1) in Hashes1:
            st.session_state.warning="This E-mail is already associated with a user"
            next_page()

        elif str(Hash1) not in Hashes1:
            engine.execute("INSERT INTO Users (Hash, email) VALUES ('{}', '{}')".format(str(Hash1), user))

            st.session_state.new_user_succesfully_signed_in = "yes"

    elif password != password_2:
        # print a different passwords error
        st.session_state.warning = "Different Passwords!"
        next_page()






