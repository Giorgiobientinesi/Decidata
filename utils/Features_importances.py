import streamlit as st
from sklearn.impute import SimpleImputer


def features_importances_Trees():
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

        #st.write(chart_data)



        fig, ax = plt.subplots()
        ax.bar(x=chart_data["Features"],height=chart_data['Importances'],color="#5b61f9")
        #plt.xticks(rotation=90)
        plt.xticks(color='purple', rotation=15, fontweight='bold', fontsize='5',
                   horizontalalignment='right')


        plt.tick_params(axis='x', colors='black', direction='out', length=2, width=1)
        #st.pyplot(fig)
        figs.append(fig)

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
        fig2, axes = plt.subplots(nrows=1, ncols=1, figsize=(8, 8), dpi=300)
        tree.plot_tree(Plot_Pipe[1],feature_names=columns_name,class_names=["True", "False"],filled=True,max_depth=3,fontsize=6)



        figs.append(fig2)

