import streamlit as st
import pandas as pd
from multiapp import MultiApp
from apps import login, train_pred_selection, dataframe_selection, columns_selection, model_selection, model_execution,trained_model_selection # import your app modules here
from utils.movements import next_page, previous_page, get_to_df_selection, get_to_var_selection,higher_priviledges_next, get_to_predict, get_to_login
import base64
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
if "user" not in st.session_state:
    st.session_state.user= "User1"
if "hash" not in st.session_state:
    st.session_state.hash = 12345
if "page_name" not in st.session_state:
    st.session_state.page_name= "login"
if "new_user_succesfully_signed_in" not in st.session_state:
    st.session_state.new_user_succesfully_signed_in="no"
######################################################################

#Multiapp Initialization

app = MultiApp()
# st.markdown("""
# # Churn Prediction App
#
# This multi-page app is automatizing the creation of Churn prediction ML models
# """)
# Add all your application here
app.add_app("Login", login.app)
#app.add_app("Train Prediction Selection",train_pred_selection.app)
app.add_app("Dataframe Selection",dataframe_selection.app)
app.add_app("Columns Selection", columns_selection.app)
app.add_app("Model Selection",model_selection.app)
app.add_app("Model Execution", model_execution.app)
app.add_app("Trained Model Selection", trained_model_selection.app)
# The main app
app.run()

#Move through the pages
if st.session_state.page_name=="Login" or st.session_state.page_name=="Train Prediction Selection":
    pass


elif st.session_state.page_name=="Model Execution":
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            Back_to_select_df = st.button("Back to Select Dataframe", on_click=get_to_df_selection)
        with col2:
            Back_to_select_var= st.button("Back to Select Variables", on_click= get_to_var_selection)

else:
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            next = st.button('Next',on_click= next_page)



#st.button("Secret just-for-developers button ;)",on_click=higher_priviledges_next)

if st.session_state.page_name!= "Login" and st.session_state.page_name!= "Train Prediction Selection":
    with st.sidebar:
        st.subheader("User:")
        st.caption(str(st.session_state.user))
        st.subheader("Go to Train")
        st.button("Go", on_click=get_to_df_selection)
        st.subheader("Go to Predict")
        st.button("Go",on_click=get_to_predict,key=2)
        st.subheader("Change User")
        st.button("Go",on_click= get_to_login,key=3)

        template = pd.read_excel("CAPSTONE-Template.xlsx")


        def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        template_csv = convert_df(template)

        st.markdown('<h3 style="color: #5b61f9;"></h3>',
                    unsafe_allow_html=True)

        st.markdown('<h3 style="color: #5b61f9;"></h3>',
                    unsafe_allow_html=True)


        st.download_button(
            "DataFrame Template ",
            template_csv,
            "template.csv",
            "text/csv",
            key='download-csv'
        )

        with open("Prediction_App-DocumentationUser.pdf", "rb") as file:
            btn=st.download_button(
            label="Documentation",
            data=file,
            file_name="Documentation.pdf",
            mime="application/octet-stream"
        )

