import streamlit as st
from utils.movements import next_page
from utils.authenticationDB import log_in_check,sign_in_user

#INITIALIZE SESSION STATE

if "light" not in st.session_state:
    st.session_state.light= "green"
if "warning" not in st.session_state:
    st.session_state.warning= "Undefined Error"
if "new_user_succesfully_signed_in" not in st.session_state:
    st.session_state.new_user_succesfully_signed_in="no"
######################################################################


def app():
    st.session_state.light="red"
    st.session_state.warning= "Please Imput E-mail and Password"
    st.markdown('<h1 style="color: #5b61f9;">Welcome!</h1>',
                unsafe_allow_html=True)
    type= st.radio("New User?",options=["Log-in","Sign-in"], index=0)
    if type == "Log-in":
        user= st.text_input("E-mail")
        password= st.text_input("Password", type="password")
        st.button("Log-in", on_click=log_in_check,args=(user,password))
    elif type== "Sign-in":
        st.session_state.warning = "Please Imput E-mail, Password and Confirm Password"
        user= st.text_input("E-mail")
        password= st.text_input("Password",type="password")
        password_2= st.text_input("Repeat Password",type="password")
        st.button("Sign-in", on_click= sign_in_user,args=(user,password,password_2))
        if st.session_state.new_user_succesfully_signed_in=="yes":
            st.success("User Succesfully Signed-in!")
            st.button("Start Here!", on_click=log_in_check, args=(user,password))
