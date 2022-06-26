#General
import streamlit as st
import pandas as pd
import math
import base64
from fpdf import FPDF
import os
import shutil
import graphviz
#Models
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.svm import  LinearSVC
#Preprocessing
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
#Metrics
from sklearn.metrics import accuracy_score,recall_score,precision_score,plot_confusion_matrix
from tempfile import NamedTemporaryFile
import matplotlib.pyplot as plt
from sklearn import tree
#Utils
from utils.splits import sort_and_drop
from utils.save_models import save_model ,create_download_button_models
import seaborn as sns
from sklearn.metrics import classification_report


#SESSION STATE INITIALIZATION
if "light" not in st.session_state:
    st.session_state.light = "red"
if "warning" not in st.session_state:
    st.session_state.warning = "Undefined Error"
if "model" not in st.session_state:
    st.session_state.model= "To_be_selected"
if 'page' not in st.session_state:
    st.session_state.page = 0
if "selected_var" not in st.session_state:
    st.session_state.selected_var=[]
if "df" not in st.session_state:
    st.session_state.df= 0
if "model_type" not in st.session_state:
    st.session_state.model_type="To_be_selected"
if "user" not in st.session_state:
    st.session_state.user= "User1"
if "hash" not in st.session_state:
    st.session_state.hash = 12345
##################################################
#IMPORT CSS
# with open('style.css') as f:
#     st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
##############################################################


#Application
def app():
    #Progress bar
    st.markdown(
        """
        <style>
            .stProgress > div > div > div > div {
                background-image: linear-gradient(to right, #5b61f9 , #5b61f9);
            }
        </style>""",
        unsafe_allow_html=True,
    )
    my_bar = st.progress(100)

    #Title and description
    st.markdown('<h1 style="color: #5b61f9;">Model results</h1>',
                unsafe_allow_html=True)
    st.write('Those are the performances of your '+ st.session_state.model)

    # Import dataframe from session state, adding month for sort the data and check if it is already there to solve strange bug

    df= st.session_state.df[st.session_state.selected_var]


    #Divided Target and Predictors

    X=df.drop(["Target"],axis=1)
    y= df["Target"]

    #identify numeric and not numeric
    categorical_var= X.select_dtypes(exclude='number').columns.tolist()
    numerical_var= X.select_dtypes(include="number").columns.tolist()

    if "Month" in categorical_var:
            categorical_var.remove("Month")
    if "Month" in numerical_var:
            numerical_var.remove("Month")


    # Top 9 classes + "other"
    for el in X[categorical_var].columns:
        if len(list(X[categorical_var][el].value_counts())) > 10:
            a = list(X[categorical_var][el].value_counts()[:10].index.tolist())
            X[el][X[categorical_var][el].isin(a) == False] = "Other"

    #Select the model to be used- Here you can change/add/delete models
    models={
        "Perceptron": MLPClassifier(solver='adam', alpha=1e-5,random_state=10),
        "SVM": LinearSVC(random_state=10),
        "Ensemble": GradientBoostingClassifier(random_state=10),
        "Logistic Regression": LogisticRegressionCV(random_state=10),
        "Simple Tree": DecisionTreeClassifier(random_state=10),
        "Random Forest": RandomForestClassifier(random_state=10)
    }
    model= models[st.session_state.model]

    #Pipeline creation
    #1) Steps
    numerical_steps=[("Imputer_num",SimpleImputer(strategy="mean"))]
    categorical_steps=[("Impupter_cat",SimpleImputer(strategy= "most_frequent")),("Encoder",OneHotEncoder(sparse=False,handle_unknown='ignore'))]


    #2) Pipes
    numerical_pipe= Pipeline(numerical_steps)
    categorical_pipe= Pipeline(categorical_steps)

    #3) Transformer
    transformer= ColumnTransformer([("Numerical Transformation",numerical_pipe,numerical_var),
                                    ("Categorical Transformation",categorical_pipe,categorical_var)],
                                   remainder="passthrough")
    Final_Pipe= Pipeline([("Transformer",transformer),("Model",model)])

    #Splitting data
    #SPLIT 1: sklearn based, stratified
    # X_train, X_test, y_train, y_test = train_test_split(X, y)
    # X_train,X_test = X_train.drop(["Month"],axis=1), X_test.drop(["Month"],axis=1)

    #SPLIT 2: Own implementation, single variable-based
    X_train, X_test, y_train, y_test= sort_and_drop(X,y,"Month")


    #Fitting model
    Final_Pipe.fit(X_train, y_train)
    predictions = Final_Pipe.predict(X_test)

    #######################################################
    #START REPORT
    ########################################################

    #General Metrics
    st.write("**Accuracy**" +": " + "Percentage of well classified rows.")
    st.write("**Precision**" +": " + "Probability that an object predicted to be true is actually true.")
    st.write("**Recall**" +": " + "Measure of how many true elements were detected.")

    #General Metrics
    st.markdown('<h2 style="color: #5B61F9;">Metrics</h2>',
                unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    classification_report_ = classification_report(y_test, predictions,output_dict=True)


    classification_report_df = pd.DataFrame(classification_report_).transpose()


    def convert_df(df):
        return df.to_csv().encode('utf-8')


    classification_report_df_csv = convert_df(classification_report_df)


    st.download_button(
        "Press to Download a more detailed metrics report! ",
        classification_report_df_csv,
        "classification_report.csv",
        "text/csv",
        key='download-csv'
    )



    col1.metric("Accuracy", str(accuracy_score(y_test,predictions))[:4])
    col2.metric("Precision", str(precision_score(y_test, predictions))[:4])
    col3.metric("Recall", str(recall_score(y_test, predictions))[:4])


    st.markdown('<h2 style="color: #5B61F9;">Confusion Matrix</h2>',
                unsafe_allow_html=True)

    a = plot_confusion_matrix(Final_Pipe, X_test, y_test,cmap="Purples")
    st.pyplot(a.figure_)






    #X = sns.boxplot(x=pd.to_numeric(df["Target"]), y=df["Customer service calls"], palette="husl", data=df)
    #st.pyplot(X.figure)

    #st.write(a)
    #st.write(X_test)

    figs= []


    figs.append(a.figure_)

    #Trees specific metrics:
    if st.session_state.model=="Random Forest" or st.session_state.model=="Simple Tree":
        st.markdown('<h2 style="color: #5B61F9;">Features Importance</h2>',
                    unsafe_allow_html=True)




        SimpleImputer.get_feature_names_out = (lambda self, names=None:
                                               self.feature_names_in_)

        columns = (Final_Pipe[:-1].get_feature_names_out())


        columns = list(columns)

        columns_name = []
        for el in columns:
            columns_name.append(el[26:])



        feature_importance = (Final_Pipe[1].feature_importances_)




        #st.write(feature_importance,index=df1.columns)

        forest_importances = pd.Series(feature_importance, index=columns_name)

        chart_data = pd.DataFrame(
            [feature_importance],
            columns=columns_name)

        chart_data = chart_data.transpose()
        Features_importance = st.bar_chart(chart_data)



        chart_data.columns = ["Importances"]
        chart_data["Features"] = list(chart_data.transpose().columns)

        chart_data = chart_data.sort_values('Importances',ascending=False)

        chart_data = chart_data.head(15)

        fig, ax = plt.subplots()
        ax.bar(x=chart_data["Features"],height=chart_data['Importances'],color="#5b61f9")
        #plt.xticks(rotation=90)
        plt.xticks(color='purple', rotation=15, fontweight='bold', fontsize='5',
                   horizontalalignment='right')


        plt.tick_params(axis='x', colors='black', direction='out', length=2, width=1)
        #st.pyplot(fig)
        figs.append(fig)






        #MOST IMPORTANT VARIABLES PLOTS

        #TARGET = 0
        first_variable = chart_data["Features"].iloc[0]
        second_variable = chart_data["Features"].iloc[1]
        third_variable = chart_data["Features"].iloc[2]








        variables_to_plot = []

        variables_to_plot.append([first_variable,second_variable,third_variable])
        target_false = df[df["Target"] == 0]



        target_false = Final_Pipe[0].transform(target_false)

        columns_transformed = (Final_Pipe[0].get_feature_names_out())
        columns_transformed = pd.DataFrame(columns_transformed,columns=["Names"])
        names_trans = list(columns_transformed["Names"])

        names_trans_correct = []
        for el in names_trans:
            names_trans_correct.append(el[26:])


        target_false = pd.DataFrame(target_false)
        target_false.columns = names_trans_correct

        for el in list(target_false.columns):
            if el[0] == "_":
                target_false[el] = target_false[el].astype(object)


        col1, col2 = st.columns(2)

        col1.markdown('<h2 style="color: #5B61F9;">Target is False</h2>',
                      unsafe_allow_html=True)





        for el in variables_to_plot[0]:
            if target_false[el].dtypes == "int64" or target_false[el].dtypes =="float64":
                fig_box = plt.figure(figsize=(10, 4))
                sns.boxplot(x=target_false[el], palette="mako", data=target_false)
                figs.append(fig_box)
                col1.pyplot(fig_box)
            elif target_false[el].dtypes == "object":
                col1.bar_chart(target_false[el].value_counts())

                fig2, ax = plt.subplots()
                ax.bar(x=target_false[el].unique(), height= target_false[el].value_counts(), color="#4c78a8")
                ax.set_xlabel(el)

                # plt.xticks(rotation=90)
                plt.xticks(color='purple', rotation=15, fontweight='bold', fontsize='5',
                           horizontalalignment='right')

                plt.tick_params(axis='x', colors='black', direction='out', length=2, width=1)
                # st.pyplot(fig)
                figs.append(fig2)

        col2.markdown('<h2 style="color: #5B61F9;">Target is True</h2>',
                      unsafe_allow_html=True)

        target_true = df[df["Target"] == 1]

        target_true = Final_Pipe[0].transform(target_true)


        target_true = pd.DataFrame(target_true)
        target_true.columns = names_trans_correct

        for el in list(target_true.columns):
            if el[0] == "_":
                target_true[el] = target_true[el].astype(object)


        for el in variables_to_plot[0]:
            if target_true[el].dtypes == "int64" or target_true[el].dtypes =="float64":
                fig_box = plt.figure(figsize=(10, 4))
                sns.boxplot(x=target_true[el], palette="mako", data=target_true)
                figs.append(fig_box)
                col2.pyplot(fig_box)

            elif target_true[el].dtypes == "object":
                col2.bar_chart(target_true[el].value_counts())

                fig2, ax = plt.subplots()
                ax.bar(x=target_true[el].unique(), height= target_true[el].value_counts(), color="#4c78a8")
                ax.set_xlabel(el)

                # plt.xticks(rotation=90)
                plt.xticks(color='purple', rotation=15, fontweight='bold', fontsize='5',
                           horizontalalignment='right')

                plt.tick_params(axis='x', colors='black', direction='out', length=2, width=1)
                # st.pyplot(fig)
                figs.append(fig2)





        #st.write(chart_data["Features"].iloc[0])

        #st.write(chart_data)





        #st.write(type(fig))



        st.markdown('<h2 style="color: #5B61F9;">Tree Plot</h2>',
                    unsafe_allow_html=True)

        Plot_Pipe = Pipeline([("Transformer", transformer), ("Model", DecisionTreeClassifier(max_depth=3))])
        Plot_Pipe.fit(X_train, y_train)


        dot_data = tree.export_graphviz(Plot_Pipe[1],feature_names=columns_name,class_names=["True", "False"],                                filled=True, rounded=True,
                                special_characters=True)


        #st.write((tree.plot_tree(Plot_Pipe[1])))

        st.graphviz_chart(dot_data)



        #st.write(graph.view())

        #THIS IS FOR THE PDF
        fig4, axes = plt.subplots(nrows=1, ncols=1, figsize=(8, 8), dpi=300)
        tree.plot_tree(Plot_Pipe[1],feature_names=columns_name,class_names=["True", "False"],filled=True,max_depth=3,fontsize=6)



        figs.append(fig4)





    elif st.session_state.model == "Logistic Regression" or st.session_state.model=="SVM":
        st.markdown('<h2 style="color: #5B61F9;">Features Importance</h2>',
                    unsafe_allow_html=True)




        SimpleImputer.get_feature_names_out = (lambda self, names=None:
                                               self.feature_names_in_)

        columns = (Final_Pipe[:-1].get_feature_names_out())


        columns = list(columns)

        columns_name = []
        for el in columns:
            columns_name.append(el[26:])


        feature_importance = (Final_Pipe[-1].coef_)

        # st.write(feature_importance,index=df1.columns)

        chart_data = pd.DataFrame(list(feature_importance.transpose()),index = columns_name)

        st.bar_chart(chart_data)


        chart_data.columns = ["Importances"]
        chart_data["Features"] = list(chart_data.transpose().columns)


        chart_data_forplots = chart_data
        chart_data = chart_data.sort_values('Importances',ascending=False)


        chart_data = chart_data.head(15)

        fig, ax = plt.subplots()
        ax.bar(x=chart_data["Features"],height=chart_data['Importances'],color="#4c78a8")
        #plt.xticks(rotation=90)
        plt.xticks(color='purple', rotation=15, fontweight='bold', fontsize='5',
                   horizontalalignment='right')


        plt.tick_params(axis='x', colors='black', direction='out', length=2, width=1)
        #st.pyplot(fig)
        figs.append(fig)


        #MOST IMPORTANT VARIABLES PLOTS

        #TARGET = 0
        first_variable = chart_data["Features"].iloc[0]
        second_variable = chart_data["Features"].iloc[1]
        third_variable = chart_data["Features"].iloc[2]








        variables_to_plot = []

        variables_to_plot.append([first_variable,second_variable,third_variable])
        target_false = df[df["Target"] == 0]


        target_false = Final_Pipe[0].transform(target_false)

        columns_transformed = (Final_Pipe[0].get_feature_names_out())
        columns_transformed = pd.DataFrame(columns_transformed,columns=["Names"])
        names_trans = list(columns_transformed["Names"])

        names_trans_correct = []
        for el in names_trans:
            names_trans_correct.append(el[26:])


        target_false = pd.DataFrame(target_false)
        target_false.columns = names_trans_correct

        for el in list(target_false.columns):
            if el[0] == "_":
                target_false[el] = target_false[el].astype(object)

        col1, col2 = st.columns(2)

        col1.markdown('<h2 style="color: #5B61F9;">Target is False</h2>',
                      unsafe_allow_html=True)




        for el in variables_to_plot[0]:
            if target_false[el].dtypes == "int64" or target_false[el].dtypes =="float64":
                fig_box = plt.figure(figsize=(10, 4))
                sns.boxplot(x=target_false[el], palette="mako", data=target_false)
                figs.append(fig_box)
                col1.pyplot(fig_box)
            elif target_false[el].dtypes == "object":
                col1.bar_chart(target_false[el].value_counts())
                fig2, ax = plt.subplots()
                ax.bar(x=target_false[el].unique(), height= target_false[el].value_counts(), color="#4c78a8")
                ax.set_xlabel(el)
                # plt.xticks(rotation=90)
                plt.xticks(color='purple', rotation=15, fontweight='bold', fontsize='5',
                           horizontalalignment='right')

                plt.tick_params(axis='x', colors='black', direction='out', length=2, width=1)
                # st.pyplot(fig)
                figs.append(fig2)


        col2.markdown('<h2 style="color: #5B61F9;">Target is True</h2>',
                      unsafe_allow_html=True)
        target_true = df[df["Target"] == 1]

        target_true = Final_Pipe[0].transform(target_true)


        target_true = pd.DataFrame(target_true)
        target_true.columns = names_trans_correct

        for el in list(target_true.columns):
            if el[0] == "_":
                target_true[el] = target_true[el].astype(object)


        for el in variables_to_plot[0]:
            if target_true[el].dtypes == "int64" or target_true[el].dtypes =="float64":
                fig_box = plt.figure(figsize=(10, 4))
                sns.boxplot(x=target_true[el], palette="mako", data=target_true)
                figs.append(fig_box)
                col2.pyplot(fig_box)

            elif target_false[el].dtypes == "object":
                col2.bar_chart(target_true[el].value_counts())

                fig2, ax = plt.subplots()
                ax.bar(x=target_true[el].unique(), height= target_true[el].value_counts(), color="#4c78a8")
                ax.set_xlabel(el)

                # plt.xticks(rotation=90)
                plt.xticks(color='purple', rotation=15, fontweight='bold', fontsize='5',
                           horizontalalignment='right')

                plt.tick_params(axis='x', colors='black', direction='out', length=2, width=1)
                # st.pyplot(fig)
                figs.append(fig2)









    elif st.session_state.model == "Ensemble":
        st.markdown('<h2 style="color: #5B61F9;">Features Importance</h2>',
                    unsafe_allow_html=True)




        SimpleImputer.get_feature_names_out = (lambda self, names=None:
                                               self.feature_names_in_)

        columns = (Final_Pipe[:-1].get_feature_names_out())


        columns = list(columns)

        columns_name = []
        for el in columns:
            columns_name.append(el[26:])



        feature_importance = (Final_Pipe[1].feature_importances_)




        #st.write(feature_importance,index=df1.columns)

        forest_importances = pd.Series(feature_importance, index=columns_name)

        chart_data = pd.DataFrame(
            [feature_importance],
            columns=columns_name)

        chart_data = chart_data.transpose()
        Features_importance = st.bar_chart(chart_data)



        chart_data.columns = ["Importances"]
        chart_data["Features"] = list(chart_data.transpose().columns)

        chart_data = chart_data.sort_values('Importances',ascending=False)

        chart_data = chart_data.head(15)


        fig, ax = plt.subplots()
        ax.bar(x=chart_data["Features"],height=chart_data['Importances'],color="#4c78a8")
        #plt.xticks(rotation=90)
        plt.xticks(color='purple', rotation=15, fontweight='bold', fontsize='5',
                   horizontalalignment='right')


        plt.tick_params(axis='x', colors='black', direction='out', length=2, width=1)
        #st.pyplot(fig)
        figs.append(fig)

        #BOXPLOTS
        #TARGET = 0
        first_variable = chart_data["Features"].iloc[0]
        second_variable = chart_data["Features"].iloc[1]
        third_variable = chart_data["Features"].iloc[2]








        variables_to_plot = []

        variables_to_plot.append([first_variable,second_variable,third_variable])
        target_false = df[df["Target"] == 0]


        target_false = Final_Pipe[0].transform(target_false)

        columns_transformed = (Final_Pipe[0].get_feature_names_out())
        columns_transformed = pd.DataFrame(columns_transformed,columns=["Names"])
        names_trans = list(columns_transformed["Names"])

        names_trans_correct = []
        for el in names_trans:
            names_trans_correct.append(el[26:])


        target_false = pd.DataFrame(target_false)
        target_false.columns = names_trans_correct

        for el in list(target_false.columns):
            if el[0] == "_":
                target_false[el] = target_false[el].astype(object)


        col1, col2 = st.columns(2)

        col1.markdown('<h2 style="color: #5B61F9;">Target is False</h2>',
                      unsafe_allow_html=True)

        for el in variables_to_plot[0]:
            if target_false[el].dtypes == "int64" or target_false[el].dtypes == "float64":
                fig_box = plt.figure(figsize=(10, 4))
                sns.boxplot(x=target_false[el], palette="mako", data=target_false)
                figs.append(fig_box)
                col1.pyplot(fig_box)
            elif target_false[el].dtypes == "object":
                col1.bar_chart(target_false[el].value_counts())
                fig_obj, ax = plt.subplots()
                ax.bar(x=target_false[el].unique(), height= target_false[el].value_counts(), color="#4c78a8")
                ax.set_xlabel(el)

                # plt.xticks(rotation=90)
                plt.xticks(color='purple', rotation=15, fontweight='bold', fontsize='5',
                           horizontalalignment='right')

                plt.tick_params(axis='x', colors='black', direction='out', length=2, width=1)
                # st.pyplot(fig)
                figs.append(fig_obj)

        col2.markdown('<h2 style="color: #5B61F9;">Target is True</h2>',
                      unsafe_allow_html=True)
        target_true = df[df["Target"] == 1]

        target_true = Final_Pipe[0].transform(target_true)


        target_true = pd.DataFrame(target_true)
        target_true.columns = names_trans_correct

        for el in list(target_true.columns):
            if el[0] == "_":
                target_true[el] = target_true[el].astype(object)

        for el in variables_to_plot[0]:
            if target_true[el].dtypes == "int64" or target_true[el].dtypes == "float64":
                fig_box = plt.figure(figsize=(10, 4))
                sns.boxplot(x=target_true[el], palette="mako", data=target_true)
                figs.append(fig_box)
                col2.pyplot(fig_box)

            elif target_false[el].dtypes == "object":
                col2.bar_chart(target_true[el].value_counts())

                fig_obj, ax = plt.subplots()
                ax.bar(x=target_true[el].unique(), height= target_true[el].value_counts(), color="#4c78a8")
                ax.set_xlabel(el)

                # plt.xticks(rotation=90)
                plt.xticks(color='purple', rotation=15, fontweight='bold', fontsize='5',
                           horizontalalignment='right')

                plt.tick_params(axis='x', colors='black', direction='out', length=2, width=1)
                # st.pyplot(fig)
                figs.append(fig_obj)

        #st.write(chart_data)


    acc = str(accuracy_score(y_test, predictions))[:4]
    prec= str(precision_score(y_test, predictions))[:4]
    rec = str(recall_score(y_test, predictions))[:4]






    import base64


    export_as_pdf = st.button("Export Report")

    def create_download_link(val, filename):
        b64 = base64.b64encode(val)  # val looks like b'...'
        return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="{filename}.pdf">Download file</a>'


    class PDF(FPDF):
        def header(self):
            #self.image("/Users/giorgiobientinesi/Desktop/background-color.png", x=0, y=0, w=210, h=297, type='',
                       #link='')

            # Rendering logo:
            #self.image("Decidata-logo.jpeg", 170, 12, 30)
            # Setting font: helvetica bold 15
            self.set_text_color(0, 0, 0)
            self.set_font('Arial', "B", 25)
            # Moving cursor to the right:
            self.cell(80)
            # Printing title:
            self.cell(30, 10, "Report", 1, 0, "C")
            # Performing a line break:
            self.ln(20)

        def footer(self):
            # Position cursor at 1.5 cm from bottom:
            self.set_y(-15)
            # Setting font: helvetica italic 8
            self.set_font('Arial', "B", 8)
            # Printing page number:
            self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", 0, 0, "C")

    # Instantiation of inherited class

    if export_as_pdf:
        pdf = PDF()
        pdf.alias_nb_pages()
        pdf.add_page()
        pdf.set_font("Times", size=12)

        pdf.set_font('Arial', "B", 22)
        pdf.set_text_color(91, 97, 249)
        pdf.cell(60, 10, "Metrics", 0, 1)

        pdf.set_text_color(0, 0, 0)

        pdf.set_font("Arial", size=12)

        pdf.cell(0, 10,"**Accuracy**: " +acc+ ". This means that " +str(float(acc)*100)[:3] + "% of the total people are well classified." , 0, 1)
        pdf.cell(0, 10,"**Precision**: " +prec+ ". When a person is classified as True, there is " +str(float(prec)*100)[:3] + "% of probability that is True." , 0, 1)
        pdf.cell(0, 10,"**Recall**: " +rec+ ". Among all of the True, " +str(float(rec)*100)[:3] + "% are found." , 0, 1)

        pdf.cell(0, 10, "", 0, 1)

        pdf.set_font('Arial', "B", 22)
        pdf.set_text_color(91, 97, 249)
        pdf.cell(60, 10, "Confusion Matrix", 0, 1,align="C")

        with NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
            figs[0].savefig(tmpfile.name)
            pdf.image(tmpfile.name, 0, 100, 180, 150)

        if len(figs)>1:
            pdf.add_page()
            pdf.cell(0, 10, "Features Importance")
            with NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
                figs[1].savefig(tmpfile.name)


                pdf.image(tmpfile.name, 5, 45, 170, 140)

            pdf.add_page()

            with NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
                pdf.cell(0, 10, "Most important features comparison")
                figs[2].savefig(tmpfile.name)


                pdf.image(tmpfile.name, 5, 40, 80, 50)

            with NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
                figs[5].savefig(tmpfile.name)

                pdf.image(tmpfile.name, 100, 40, 80, 50)




            with NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
                figs[3].savefig(tmpfile.name)

                pdf.image(tmpfile.name, 5, 120, 80, 50)

            with NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
                figs[6].savefig(tmpfile.name)

                pdf.image(tmpfile.name, 100, 120, 80, 50)


            with NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
                figs[4].savefig(tmpfile.name)

                pdf.image(tmpfile.name, 5, 200, 80, 50)

            with NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
                figs[7].savefig(tmpfile.name)

                pdf.image(tmpfile.name, 100, 200, 80, 50)



            if len(figs)>8:
                pdf.add_page()

                pdf.cell(0, 10, "Tree")
                with NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
                    figs[-1].savefig(tmpfile.name)

                    pdf.image(tmpfile.name, 5, 45, 200, 150)



        html = create_download_link(pdf.output(dest="S").encode("latin-1"), "Report")

        st.markdown(html, unsafe_allow_html=True)


    st.markdown('<h2 style="color: #ffffff";">Space here</h2>',
                unsafe_allow_html=True)

    #Save model button
    save=st.button("Save your model")
    if save:
        save_model(Final_Pipe)
        st.success('Model Correctly Saved!')


    #Download Folder
    create_download_button_models()

