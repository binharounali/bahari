#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 31 19:03:05 2020

@author: HarounJr
"""

def datasplit(inputs,x,y):
    from sklearn.model_selection import train_test_split

    sp = inputs.at[0,'Split']
    if sp == "Yes":
        j = inputs.at[0,'TestSize']
        i = int(inputs.at[0,'RandomState'])
        k = inputs.at[0,'StratifiedSampling']
        if k == "Yes":
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=j, random_state = i, stratify = y)
        elif k == "No":
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=j, random_state = i)
    else:
        x_train = x
        y_train = y
        x_test =  x
        y_test = y
    
    return x_train, x_test, y_train, y_test

def impute(inputs,x,y):
    from sklearn.impute import SimpleImputer
    import pandas as pd
    i = inputs.at[0,'Strategy']
    if i == "Most_Frequent":
        col_name = x.columns.values.tolist()
        imputer = SimpleImputer(strategy = "most_frequent")
        x = imputer.fit_transform(x)
        y = imputer.fit_transform(y)
        x = pd.DataFrame(x)
        x.columns = col_name
    elif i == "Mean":
        imputer = SimpleImputer(strategy = "mean")
        x = imputer.fit_transform(x)
        y = imputer.fit_transform(y)
    elif i == "Median":
        imputer = SimpleImputer(strategy = "median")
        x = imputer.fit_transform(x)
        y = imputer.fit_transform(y)
    elif i == "Constant":
#        i = int(input("Enter the Constant: ")) 
        imputer = SimpleImputer(strategy = "constant", fill_value = 0 )
        x = imputer.fit_transform(x)
        y = imputer.fit_transform(y)
    else:
        print("unknown action, Please re-run the app and select only above-mentioned options") 
    return x, y

def scale(inputs,x,y):
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import MinMaxScaler
    import pandas as pd
    i = inputs.at[0,'ScaleTechnique']
    if i == "Standardize":
        columns = x.columns.values.tolist()
        scaler = StandardScaler()
        x = scaler.fit_transform(x)
        x = pd.DataFrame(x, columns = columns)
        x.columns = columns
    elif i == "Normalize":
        columns = x.columns.values.tolist()
        norm = MinMaxScaler()
        x = norm.fit_transform(x)
        x = pd.DataFrame(x, columns = columns)
        x.columns = columns
    else:
        print("unknown action, Please re-run the app and select only above-mentioned options") 
    return x, y