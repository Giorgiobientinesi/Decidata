import streamlit as st
import pandas as pd

#SESSION STATE INITIALIZATION
if 'df' not in st.session_state:
    st.session_state.df = 0
if "light" not in st.session_state:
    st.session_state.light = "red"
if "warning" not in st.session_state:
    st.session_state.warning = "Undefined Error"
if "selected_var" not in st.session_state:
    st.session_state.selected_var=[]


def app():
    st.session_state.light = "red"
    st.session_state.warning = "Please select at least one predictor."

    st.markdown(
        """
        <style>
            .stProgress > div > div > div > div {
                background-image: linear-gradient(to right, #5b61f9 , #5b61f9);
            }
        </style>""",
        unsafe_allow_html=True,
    )
    my_bar = st.progress(50)

    if 'df' not in st.session_state:
        st.session_state.df = 0

    df = st.session_state.df
    df1 = st.session_state.df
    st.markdown('<h1 style="color: #5b61f9;">Null Values Detection</h1>',
                                 unsafe_allow_html=True)

    st.write("These are the percentage of Null Values in your Dataset. Please Select the variables you want to include in the analysis!")

    variables = []



    All = st.checkbox("Select All", key=0)
    if All == True:
        variables.append(list(df.columns))

        variables = variables[0]




        percent_missing = df.isnull().sum() * 100 / len(df)
        missing_values = pd.DataFrame({"percent_missing": percent_missing})
        cols = st.columns(3)

        i = 1
        if len(df.columns) % 2 == 0:
            k = len(df.columns) / 2
            k = int(k)
        else:
            k = len(df.columns)
            k = k - 1
            k = k / 2
            k = int(k)

        for el in list(df1.columns[:k]):

            with st.container():
                percent_missing = str(df1[el].isnull().sum() * 100 / len(df1))

                for j in ["Target","Customer ID","Month"]:
                    if el ==j:
                        cols[2].markdown('<h6 style="color: #5b61f9;">{}</h6>'.format(str(el) + " " + percent_missing[0:4] + "%"),
                                         unsafe_allow_html=True)

                        a = cols[2].checkbox("", key=i,disabled=True,value=True)
                        df1 = df1.drop([el],axis=1)



                if float(percent_missing) == 0:
                    if el not in ["Target","Customer ID","Month"]:


                        cols[0].markdown('<h6 style="color: green;">{}</h6>'.format(str(el) + " " + percent_missing[0:4] + "%"),
                                         unsafe_allow_html=True)

                        a = cols[0].checkbox("", key=i,disabled=True,value=True)


                elif float(percent_missing) < 20 and float(percent_missing) > 0:
                    if el not in ["Target", "Customer ID", "Month"]:
                        cols[1].markdown('<h6 style="color: orange;">{}</h6>'.format(str(el) + " " + percent_missing[0:4] + "%"),
                            unsafe_allow_html=True)
                        a = cols[1].checkbox("", key=i,disabled=True,value=True)

                else:
                    if el not in ["Target", "Customer ID", "Month"]:
                        cols[2].markdown('<h6 style="color: red;">{}</h6>'.format(str(el) + " " + percent_missing[0:4] + "%"),
                                            unsafe_allow_html=True)
                        a = cols[2].checkbox("", key=i,disabled=True,value=True)

            i += 1

        for el in list(df1.columns[k:]):

            with st.container():
                percent_missing = str(df1[el].isnull().sum() * 100 / len(df1))

                for j in ["Target","Customer ID","Month"]:
                    if el == j:
                        cols[2].markdown('<h6 style="color: #5b61f9;">{}</h6>'.format(str(el) + " " + percent_missing[0:4] + "%"),
                                         unsafe_allow_html=True)
                        a = cols[2].checkbox("", key=i,disabled=True,value=True)
                        df1 = df1.drop([el],axis=1)

                if float(percent_missing) == 0:
                    if el not in ["Target","Customer ID","Month"]:
                        cols[1].markdown('<h6 style="color: green;">{}</h6>'.format(str(el) + " " + percent_missing[0:4] + "%"),
                                         unsafe_allow_html=True)

                        a = cols[1].checkbox("", key=i,disabled=True,value=True)

                elif float(percent_missing) < 20 and float(percent_missing) > 0:
                    if el not in ["Target", "Customer ID", "Month"]:
                        cols[0].markdown(
                            '<h6 style="color: orange;">{}</h6>'.format(str(el) + " " + percent_missing[0:4] + "%"),
                            unsafe_allow_html=True)
                        a = cols[0].checkbox("", key=i,disabled=True,value=True)

                else:
                    if el not in ["Target", "Customer ID", "Month"]:
                        cols[2].markdown('<h6 style="color: red;">{}</h6>'.format(str(el) + " " + percent_missing[0:4] + "%"),
                                         unsafe_allow_html=True)
                        a = cols[2].checkbox("", key=i,disabled=True,value=True)


            i += 1


    else:
        percent_missing = df.isnull().sum() * 100 / len(df)
        missing_values = pd.DataFrame({"percent_missing": percent_missing})
        variables = []
        cols = st.columns(3)

        i = 1
        if len(df.columns) % 2 == 0:
            k = len(df.columns) / 2
            k = int(k)
        else:
            k = len(df.columns)
            k = k - 1
            k = k / 2
            k = int(k)

        for el in list(df1.columns[:k]):

            with st.container():
                percent_missing = str(df1[el].isnull().sum() * 100 / len(df1))

                for j in ["Target", "Customer ID", "Month"]:
                    if el == j:
                        cols[2].markdown(
                            '<h6 style="color: #5b61f9;">{}</h6>'.format(str(el) + " " + percent_missing[0:4] + "%"),
                            unsafe_allow_html=True)

                        a = cols[2].checkbox("", key=i, disabled=True, value=True)
                        df1 = df1.drop([el], axis=1)
                        variables.append(el)

                if float(percent_missing) == 0:
                    if el not in ["Target", "Customer ID", "Month"]:

                        cols[0].markdown(
                            '<h6 style="color: green;">{}</h6>'.format(str(el) + " " + percent_missing[0:4] + "%"),
                            unsafe_allow_html=True)

                        a = cols[0].checkbox("", key=i)
                        if a == True:
                            variables.append(el)

                elif float(percent_missing) < 20 and float(percent_missing) > 0:
                    if el not in ["Target", "Customer ID", "Month"]:
                        cols[1].markdown(
                            '<h6 style="color: orange;">{}</h6>'.format(str(el) + " " + percent_missing[0:4] + "%"),
                            unsafe_allow_html=True)
                        a = cols[1].checkbox("", key=i)
                        if a == True:
                            variables.append(el)
                else:
                    if el not in ["Target", "Customer ID", "Month"]:
                        cols[2].markdown(
                            '<h6 style="color: red;">{}</h6>'.format(str(el) + " " + percent_missing[0:4] + "%"),
                            unsafe_allow_html=True)
                        a = cols[2].checkbox("", key=i)
                        if a == True:
                            variables.append(el)

            i += 1

        for el in list(df1.columns[k:]):

            with st.container():
                percent_missing = str(df1[el].isnull().sum() * 100 / len(df1))

                for j in ["Target", "Customer ID", "Month"]:
                    if el == j:
                        cols[2].markdown(
                            '<h6 style="color: #5b61f9;">{}</h6>'.format(str(el) + " " + percent_missing[0:4] + "%"),
                            unsafe_allow_html=True)
                        a = cols[2].checkbox("", key=i, disabled=True, value=True)
                        df1 = df1.drop([el], axis=1)
                        variables.append(el)

                if float(percent_missing) == 0:
                    if el not in ["Target", "Customer ID", "Month"]:
                        cols[1].markdown(
                            '<h6 style="color: green;">{}</h6>'.format(str(el) + " " + percent_missing[0:4] + "%"),
                            unsafe_allow_html=True)

                        a = cols[1].checkbox("", key=i)
                        if a == True:
                            variables.append(el)
                elif float(percent_missing) < 20 and float(percent_missing) > 0:
                    if el not in ["Target", "Customer ID", "Month"]:
                        cols[0].markdown(
                            '<h6 style="color: orange;">{}</h6>'.format(str(el) + " " + percent_missing[0:4] + "%"),
                            unsafe_allow_html=True)
                        a = cols[0].checkbox("", key=i)
                        if a == True:
                            variables.append(el)
                else:
                    if el not in ["Target", "Customer ID", "Month"]:
                        cols[2].markdown(
                            '<h6 style="color: red;">{}</h6>'.format(str(el) + " " + percent_missing[0:4] + "%"),
                            unsafe_allow_html=True)
                        a = cols[2].checkbox("", key=i)
                        if a == True:
                            variables.append(el)

            i += 1







    variables.remove("Month")
    variables.remove("Customer ID")
    st.session_state.selected_var= variables
    df = df[st.session_state.selected_var]
    st.dataframe(df)


    if len(variables)>1:
        st.session_state.light = "green"

    #Passing list of selected var to next step
