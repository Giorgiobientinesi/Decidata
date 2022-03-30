#General
import streamlit as st
import pandas as pd
import math
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
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score,recall_score,precision_score
from tempfile import NamedTemporaryFile
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import base64
from sklearn.metrics import plot_confusion_matrix
from fpdf import FPDF


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

##################################################
#IMPORT CSS
#with open('style.css') as f:
    #st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
####################################################
#Train-test Split by date
#Assumes the column Target to exist
def train_test_by_var(X,y,var):
    df= pd.concat([X,y],axis=1)
    df.sort_values(by=var,inplace=True)
    y = df["Target"]
    X= df.drop(["Target"],axis=1)
    train_instances= int(math.modf((X.shape[0]/100)*80)[1])
    return X.iloc[0:train_instances,:], X.iloc[train_instances: ,:] , y[0:train_instances], y[train_instances :]



    if Metrics_explanation:
        st.write("**Accuracy**")

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
    st.write('Those are the performances of your Model.')

    #Metrics_explanation = st.button("Find More about Metrics")


    st.write("**Accuracy**" +": " + "Percentage of well classified rows.")
    st.write("**Precision**" +": " + "Probability that an object predicted to be true is actually true.")
    st.write("**Recall**" +": " + "Measure of how many true elements were detected.")



    # Import dataframe from session state, adding month for sort the data and check if it is already there to solve strange bug
    to_keep= st.session_state.selected_var
    if "Month" not in to_keep:
        to_keep.append("Month")
    df= st.session_state.df[to_keep]

    #Divided Target and Predictors

    X=df.drop(["Target"],axis=1)
    y= df["Target"]


    #identify numeric and not numeric
    categorical_var= X.select_dtypes(exclude='number').columns.tolist()#   include other datatypes
    numerical_var= X.select_dtypes(include="number").columns.tolist()

    if "Month" in categorical_var:
        categorical_var.remove("Month")
    if "Month" in categorical_var:
        numerical_var.remove("Month")


    # Top 9 classes + "other"
    for el in X[categorical_var].columns:
        if len(list(X[categorical_var][el].value_counts())) > 10:
            a = list(X[categorical_var][el].value_counts()[:10].index.tolist())
            X[el][X[categorical_var][el].isin(a) == False] = "Other"


    #Select the model to be used- Here you can change/add/delete models
    models={
        "Perceptron": MLPClassifier(solver='adam', alpha=1e-5),
        "SVM": LinearSVC(),
        "Ensemble": GradientBoostingClassifier(),
        "Logistic Regression": LogisticRegressionCV(),
        "Simple Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier()
    }
    model= models[st.session_state.model]

    #Pipeline creation
    #1) Steps
    numerical_steps=[("Imputer_num",SimpleImputer(strategy="mean"))]
    categorical_steps=[("Impupter_cat",SimpleImputer(strategy= "most_frequent")),("Encoder",OneHotEncoder(sparse=False))]


    #2) Pipes
    numerical_pipe= Pipeline(numerical_steps)
    categorical_pipe= Pipeline(categorical_steps)

    #3) Transformer
    transformer= ColumnTransformer([("Numerical Transformation",numerical_pipe,numerical_var),
                                    ("Categorical Transformation",categorical_pipe,categorical_var)],
                                   remainder="passthrough")
    Final_Pipe= Pipeline([("Transformer",transformer),("Model",model)])

    #Splitting data
    #X_train, X_test, y_train, y_test = train_test_split(X, y)

    X_train, X_test, y_train, y_test= train_test_by_var(X,y,"Month")

    X_train,X_test = X_train.drop(["Month"],axis=1), X_test.drop(["Month"],axis=1)

    #Fitting model
    Final_Pipe.fit(X_train, y_train)
    predictions = Final_Pipe.predict(X_test)

    #######################################################
    #START REPORT
    ########################################################

    #General Metrics
    st.markdown('<h2 style="color: #5B61F9;">Metrics</h2>',
                unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    col1.metric("Accuracy", str(accuracy_score(y_test,predictions))[:4])
    col2.metric("Precision", str(precision_score(y_test, predictions))[:4])
    col3.metric("Recall", str(recall_score(y_test, predictions))[:4])


    st.markdown('<h2 style="color: #5B61F9;">Confusion Matrix</h2>',
                unsafe_allow_html=True)

    a = plot_confusion_matrix(Final_Pipe, X_test, y_test,cmap="Purples")
    st.pyplot(a.figure_)

    figs=[]
    figs.append(a.figure_)
    #Trees specific metrics:
    if st.session_state.model=="Random Forest" or st.session_state.model=="Simple Tree":
        st.markdown('<h2 style="color: #5B61F9;">Features Importance</h2>',
                    unsafe_allow_html=True)
        df1 = pd.get_dummies(X_train)
        feature_importance = (Final_Pipe[1].feature_importances_)

        forest_importances = pd.Series(feature_importance, index=df1.columns)

        chart_data = pd.DataFrame(
            [feature_importance],
            columns=df1.columns)

        chart_data = chart_data.transpose()
        Features_importance = st.bar_chart(chart_data)



        chart_data.columns = ["Importances"]
        chart_data["Features"] = list(chart_data.transpose().columns)

        chart_data = chart_data.sort_values('Importances',ascending=False)

        chart_data = chart_data.head(15)

        #st.write(chart_data)



        fig, ax = plt.subplots()
        ax.bar(x=chart_data["Features"],height=chart_data['Importances'],color="#5b61f9")
        #plt.xticks(rotation=90)
        plt.xticks(color='purple', rotation=15, fontweight='bold', fontsize='5',
                   horizontalalignment='right')


        plt.tick_params(axis='x', colors='black', direction='out', length=2, width=1)
        #st.pyplot(fig)
        figs.append(fig)



        st.markdown('<h2 style="color: #5B61F9;">Tree Plot</h2>',
                    unsafe_allow_html=True)
        Plot_Pipe = Pipeline([("Transformer", transformer), ("Model", DecisionTreeClassifier(max_depth=3))])
        Plot_Pipe.fit(X_train, y_train)
        fig_tree = plt.figure()
        plot_tree(Plot_Pipe[1], feature_names=df1.columns, impurity=False, class_names=["True", "False"])
        st.pyplot(fig_tree)

        figs.append(fig_tree)












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
            self.image("Decidata-logo.jpeg", 170, 12, 30)
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

        pdf.cell(0, 10,"**Accuracy**: " +acc+ ". This means that " +str(float(acc)*100) + "% of the total people are well classified." , 0, 1)
        pdf.cell(0, 10,"**Precision**: " +prec+ ". When a person is classified as True, there is " +str(float(prec)*100) + "% of probability that is True." , 0, 1)
        pdf.cell(0, 10,"**Recall**: " +rec+ ". Among all of the True, " +str(float(rec)*100) + "% are found." , 0, 1)

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


                pdf.image(tmpfile.name, 5, 45, 180, 150)

            pdf.add_page()

            pdf.cell(0, 10, "Tree")
            with NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
                figs[2].savefig(tmpfile.name)

                pdf.image(tmpfile.name, 5, 45, 200, 150)



        html = create_download_link(pdf.output(dest="S").encode("latin-1"), "test")

        st.markdown(html, unsafe_allow_html=True)

    st.markdown('<h2 style="color: #ffffff";">Space here</h2>',
                unsafe_allow_html=True)




