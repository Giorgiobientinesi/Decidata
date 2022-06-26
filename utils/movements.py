import streamlit as st

#INITIALIZE SESSION STATE

if 'page' not in st.session_state:
    st.session_state.page = 0
if "light" not in st.session_state:
    st.session_state.light= "green"
if "warning" not in st.session_state:
    st.session_state.warning= "Undefined Error"
if "model" not in st.session_state:
    st.session_state.model = "To_be_selected"
if "model_type" not in st.session_state:
    st.session_state.model_type="To_be_selected"
######################################################################

def next_page():
    if st.session_state.light=="green":
        st.session_state.page += 1
    elif st.session_state.light=="red":
        st.warning(st.session_state.warning)

def previous_page():
    st.session_state.page-=1 #TO BE REMOVED
    st.session_state.model = "To_be_selected"
    st.session_state.model_type = "To_be_selected"

def get_to_df_selection():
    st.session_state.page=1
    st.session_state.model="To_be_selected"
    st.session_state.model_type= "To_be_selected"

def get_to_var_selection():
    st.session_state.page = 2
    st.session_state.model = "To_be_selected"
    st.session_state.model_type = "To_be_selected"

def get_to_login():
    for key in st.session_state.keys():
        del st.session_state[key]
    st.session_state.page=0

def get_to_predict():
    st.session_state.page=5

def higher_priviledges_next():
    st.session_state.light="green"
    st.session_state.page+=1

