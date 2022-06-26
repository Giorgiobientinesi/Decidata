import pandas as pd
import math
import streamlit as st

#Train-test Split by a var that will be dropped. Useful for sorting with zero variance variables like dates or id.
#Assumes the column Target to exist.
def sort_and_drop(X,y,var):
    df= pd.concat([X,y],axis=1)
    df.sort_values(by=var,inplace=True)
    df.drop([var],axis=1,inplace=True)
    y = df["Target"]
    X= df.drop(["Target"],axis=1)
    train_instances= int(math.modf((X.shape[0]/100)*80)[1])
    X_train, X_test, y_train, y_test= X.iloc[0:train_instances,:], X.iloc[train_instances: ,:] , y[0:train_instances], y[train_instances :]
    #START VERIFICATION OF CLASSES
    #Iterate over categorical columns
    for el in X_train.select_dtypes(exclude='number').columns.tolist():
        train_classes= X_train[el].value_counts().index.tolist()
        test_classes= X_test[el].value_counts().index.tolist()
        #Identify differences in the number of classes between train and test
        if len(list(set(train_classes)-set(test_classes)))>0:
            #Identify different elements
            diff= list(set(train_classes)-set(test_classes))
            for item in diff:
                #Create the new row
                dict={}
                for col in X_train.columns:
                    #if categorical
                    if col in X_train.select_dtypes(exclude='number').columns.tolist():
                        dict[col]=X_train[col].mode()
                    #if numeric
                    elif col in X_train.select_dtypes(include ='number').columns.tolist():
                        dict[col] = X_train[col].mean()
                dict[el]=str(item)
                to_add=pd.DataFrame(data=dict)
                X_test = pd.concat([X_test,to_add],axis=0)
                y_test=pd.concat([y_test,y_test.mode()])
        #Same the other way around
        if len(list(set(test_classes) - set(train_classes)))>0:
            diff = list(set(test_classes) - set(train_classes))
            for item in diff:
                #Create the new row
                dict={}
                for col in X_train.columns:
                    #if categorical use mode
                    if col in X_train.select_dtypes(exclude='number').columns.tolist():
                        dict[col]=X_train[col].mode()
                    #if numerical use avg
                    elif col in X_train.select_dtypes(include ='number').columns.tolist():
                        dict[col] = X_train[col].mean()
                dict[el]=str(item)
                to_add=pd.DataFrame(data=dict)
                X_train = pd.concat([X_train,to_add],axis=0)
                y_train = pd.concat([y_train,y_train.mode()])
    return X_train,X_test,y_train,y_test