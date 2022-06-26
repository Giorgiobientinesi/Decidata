import streamlit as st
import os

#SESSION STATE INITIALIZATION
if "light" not in st.session_state:
    st.session_state.light = "red"
if "warning" not in st.session_state:
    st.session_state.warning = "Undefined Error"
if "model" not in st.session_state:
    st.session_state.model= "To_be_selected"
if 'page' not in st.session_state:
    st.session_state.page = 0
if "model_type" not in st.session_state:
    st.session_state.model_type="To_be_selected"

####################################################
#BUTTON FUNCTIONS

#Select Model Type
def select_black():
    st.session_state.model_type = "black"
def select_white():
    st.session_state.model_type = "white"

#Select Model
def select_model(model):
    st.session_state.model=str(model)
    st.session_state.light="green"

#Build the selection page
def selection_page_builder(model_list,descriptions):
    cols= st.columns(len(model_list))
    for el in range(len(cols)):
        with cols[el]:
            with st.container():
                st.markdown('<h3 style="color: #000000;">{}</h3>'.format(model_list[el]),
                            unsafe_allow_html=True)
                st.write(descriptions[model_list[el]])
    with st.container():
        st.write("    ")
    cols2= st.columns(len(model_list))
    for el in range(len(cols2)):
        with cols2[el]:
            if st.button(str(model_list[el])):
                select_model(model_list[el])

##########################################################
#SELECTION FUNCTION
###############################################
def app():

    #Progress Bar
    st.markdown(
        """
        <style>
            .stProgress > div > div > div > div {
                background-image: linear-gradient(to right, #5b61f9 , #5b61f9);
            }
        </style>""",
        unsafe_allow_html=True,
    )
    my_bar = st.progress(75)

    #Set Light = red : the user must select a model before leaving the page
    st.session_state.warning= "You must select a model before going on with the execution!"
    if st.session_state.model == "To_be_selected":
        st.session_state.light="red"

    #Title and description
    st.markdown('<h1 style="color: #5b61f9;">Algorithm Selection</h1>',
                unsafe_allow_html=True)

    st.markdown('<h5 style="color: #ffffff;">Please select the type of algorithm you want to use!</h5>',
                unsafe_allow_html=True)

    #Part 1: Select the type of model
    if st.session_state.model_type == "To_be_selected":
        col1,col2= st.columns(2)
        with col1:
            st.markdown('<h3 style="color: #5b61f9;">Black Box Models</h2>',
                        unsafe_allow_html=True)

            st.caption(
                "Those are models with high Interpretability and limited Performances. Use them when you are interested in gaining insights!")

            w_dir = os.getcwd()
            #st.image(w_dir + "Black-box.png")


            st.button("Select", key="RF",on_click=select_black,help=("Great performances but difficult to interpret"))

        with col2:
            st.markdown('<h3 style="color: #5b61f9;">Whitebox Models</h2>',
                        unsafe_allow_html=True)

            st.caption(
                "Those are models with high Performances and limited Interpretability. Use them when you are interested in predictctions!")

            #st.image(w_dir + "White-box.png")


            st.button('Select',key="ST",on_click=select_white, help=("Easy to Interpret but lower performances"))

    #Part 2: Select the Actual Model
    #Those lists are the only thing you need to modify to change the generated page.
    #For each new element in the list remember to add the relative model in the model_execution app
    ###########################################
    black=["Perceptron","SVM","Ensemble"]
    white=["Logistic Regression","Simple Tree","Random Forest"]
    descriptions={
        "Perceptron":"The simplest neural network. This model has the ability of capturing complex relationships between your variables and achieve amazing performances.",
        "SVM": "A linear model that can capture highly nonlinear relationships.",
        "Ensemble": "Optimized tree-based model that uses the Boostin technique to improve its performances.",
        "Logistic Regression": "The simplest classification model. Very interpretable thanks to its linearity, not very good for performances.",
        "Simple Tree": "The simplest tree-based model. Fit a single decision tree for classification.",
        "Random Forest": "Improved tree-based model. Train many trees and predict base on the average."
    }

    if st.session_state.model_type=="white":
        st.markdown('<h2 style="color: #5b61f9;">White Box Models</h2>',
                    unsafe_allow_html=True)
        st.caption("Those are models with high Interpretability and limited Performances. Use them when you are interested in gaining insights!")
        selection_page_builder(white,descriptions)

    elif st.session_state.model_type=="black":
        st.markdown('<h2 style="color: #5b61f9;">Black Box Models</h2>',
                    unsafe_allow_html=True)
        st.caption("Those are models with high Performances and limited Interpretability. Use them when you are interested in predictctions!")
        selection_page_builder(black,descriptions)


    if st.session_state.model!= "To_be_selected":
        towrite= "The selected model is "+ st.session_state.model
        st.markdown('<h5 style="color: #5b61f9;">'+ towrite + '</h5>',
                unsafe_allow_html=True)


    st.markdown('<h2 style="color: #ffffff";">Space here</h2>',
                unsafe_allow_html=True)



