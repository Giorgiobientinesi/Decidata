import streamlit as st
from utils.movements import next_page, get_to_df_selection, get_to_predict

def app():
    st.markdown('<h1 style="color: #5b61f9;">Train or Predict?</h1>',
                unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.header("Train")
        st.button("Go to Train", on_click=get_to_df_selection)
    with col2:
        st.header("Predict")
        st.button("Go to Predict",on_click= get_to_predict)