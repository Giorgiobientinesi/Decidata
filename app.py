import streamlit as st
import pandas as pd
from multiapp import MultiApp
from apps import home, data, model_selection, model_execution # import your app modules here

#INITIALIZE SESSION STATE
if "selected_var" not in st.session_state:
    st.session_state.selected_var=[]
if 'page' not in st.session_state:
    st.session_state.page = 0
if "light" not in st.session_state:
    st.session_state.light= "green"
if "warning" not in st.session_state:
    st.session_state.warning= "Undefined Error"
if "model" not in st.session_state:
    st.session_state.model = "To_be_selected"
if 'df' not in st.session_state:
    st.session_state.data = 0
if "model_type" not in st.session_state:
    st.session_state.model_type="To_be_selected"
######################################################################

#Navigating through pages
def netx_page():
    if st.session_state.light=="green":
        st.session_state.page += 1
    elif st.session_state.light=="red":
        st.warning(st.session_state.warning)

def previous_page():
    st.session_state.page-=1 #TO BE REMOVED
    st.session_state.model = "To_be_selected"
    st.session_state.model_type = "To_be_selected"

def get_to_df_selection():
    st.session_state.page=0
    st.session_state.model="To_be_selected"
    st.session_state.model_type= "To_be_selected"

def get_to_var_selection():
    st.session_state.page = 1
    st.session_state.model = "To_be_selected"
    st.session_state.model_type = "To_be_selected"
########################################################################################################################
#Temporary variables / Uncomment if problem happens to see background
#st.write(st.session_state.page)
#st.write(st.session_state.light)
########################################################################################################################

#Multiapp Initialization

app = MultiApp()
# st.markdown("""
# # Churn Prediction App
#
# This multi-page app is automatizing the creation of Churn prediction ML models
# """)
# Add all your application here
app.add_app("Home", home.app)
app.add_app("Data", data.app)
app.add_app("Model Selection",model_selection.app)
app.add_app("Model Execution", model_execution.app)
# The main app
app.run()

#Move through the pages
if st.session_state.page < len(app.apps)-1:
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            next = st.button('Next',on_click= netx_page)
prev = st.button("Previous", on_click=previous_page)

if st.session_state.page== len(app.apps)-1:
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            Back_to_select_df = st.button("Back to Select Dataframe", on_click=get_to_df_selection)
        with col2:
            Back_to_select_var= st.button("Back to Select Variables", on_click= get_to_var_selection)


#st.sidebar.image("Decidata-logo.jpeg")
