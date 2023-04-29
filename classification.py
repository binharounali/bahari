#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 30 21:17:05 2020

@author: HarounJr
"""

def rf(inputs, x_train, y_train, x_test, y_test):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_validate
    import perfmetric as pm
    import pandas as pd
    
    seed = int(inputs.at[0,'RandomState'])
    if seed != 0:
        r_state = seed
    else:
        r_state = None

    folds = int(inputs.at[0,'CVFolds'])
    
    i = inputs.at[0,'Hyperparameters']
    if i == "DefaultValues":
        model = RandomForestClassifier(random_state = r_state)
    if i == "Customized":
        ne = int(inputs.at[0,'rf_customized_ne_cl'])
        msl = int(inputs.at[0,'rf_customized_msl_cl'])
        md = int(inputs.at[0,'rf_customized_md_cl'])
        model = RandomForestClassifier(max_depth = md, min_samples_leaf = msl, n_estimators = ne, random_state = r_state)
        
    cv_results_ = cross_validate(model, x_train, y_train, scoring = ['accuracy','recall_weighted','precision_weighted','f1_weighted'], cv = folds)
    model.fit(x_train,y_train)
    
    train_accuracy = (cv_results_['test_accuracy'].mean()) 
    train_recall = (cv_results_['test_recall_weighted'].mean()) 
    train_precision = (cv_results_['test_precision_weighted'].mean()) 
    train_f1 = (cv_results_['test_f1_weighted'].mean()) 
    
    feat_importances = pd.Series(model.feature_importances_, index=x_train.columns)
    dependent_var = inputs.at[0,'Y']
    feat_importances.nlargest(20).plot(kind='barh', title = "Feature Importance for preciting " + dependent_var + " using Random Forest Model")
    
    j = inputs.at[0,'ModelOptions']
    if j == "None":
        train_accuracy, train_recall,train_precision, train_f1, test_accuracy, test_recall, test_precision, test_f1, test_time = pm.perfcl(inputs, model,train_accuracy,train_recall,train_precision, train_f1, x_train,y_train,x_test,y_test)
    if j == "Test with New Data":
        train_accuracy, train_recall,train_precision, train_f1, test_accuracy, test_recall, test_precision, test_f1 = pm.perfclnew(inputs,model,train_accuracy, train_recall,train_precision, train_f1)
    
    model_name = inputs.at[0,'Methods']
    parameters = model.get_params()
    
    return parameters, model_name, train_accuracy, train_recall,train_precision, train_f1, test_accuracy, test_recall, test_precision, test_f1, test_time, feat_importances
    
def svm(inputs, x_train, y_train, x_test, y_test):
    from sklearn import svm
    from sklearn.model_selection import cross_validate
    import perfmetric as pm
    import FeatureImportance as FI
    
    seed = int(inputs.at[0,'RandomState'])
    if seed != 0:
        r_state = seed
    else:
        r_state = None

    folds = int(inputs.at[0,'CVFolds'])
    
    i = inputs.at[0,'Hyperparameters']
    if i == "DefaultValues":
        model = svm.SVC(random_state = r_state)
        kernel = 'rbf'
    if i == "Customized":
        kernel = inputs.at[0,'svm_customized_kernel_cl']
        c = inputs.at[0,'svm_customized_c_cl']
        model = svm.SVC(kernel = kernel, C = c,random_state = r_state)  
        
    cv_results_ = cross_validate(model, x_train, y_train, scoring = ['accuracy','recall_weighted','precision_weighted','f1_weighted'], cv = folds)
    model.fit(x_train,y_train)
    train_accuracy = (cv_results_['test_accuracy'].mean()) 
    train_recall = (cv_results_['test_recall_weighted'].mean()) 
    train_precision = (cv_results_['test_precision_weighted'].mean()) 
    train_f1 = (cv_results_['test_f1_weighted'].mean()) 
   
    if kernel == "linear":
        coef = model.coef_[0]
        feature_names = x_train.columns
        FI.f_importances(abs(coef), feature_names)
    else:
        coef = "Non-linear kernel, hence no FI"
        
    j = inputs.at[0,'ModelOptions']
    if j == "None":
        train_accuracy, train_recall,train_precision, train_f1, test_accuracy, test_recall, test_precision, test_f1, test_time = pm.perfcl(inputs, model,train_accuracy,train_recall,train_precision, train_f1, x_train,y_train,x_test,y_test)
    if j == "Test with New Data":
        train_accuracy, train_recall,train_precision, train_f1, test_accuracy, test_recall, test_precision, test_f1 = pm.perfclnew(inputs,model,train_accuracy, train_recall,train_precision, train_f1)
    
    model_name = inputs.at[0,'Methods']
    parameters = model.get_params()
    
    return parameters, model_name, train_accuracy, train_recall,train_precision, train_f1, test_accuracy, test_recall, test_precision, test_f1, test_time, coef

def logisticReg(inputs, x_train, y_train, x_test, y_test):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_validate
    import perfmetric as pm
    import FeatureImportance as FI
    import numpy as np

    
    seed = int(inputs.at[0,'RandomState'])
    if seed != 0:
        r_state = seed
    else:
        r_state = None

    folds = int(inputs.at[0,'CVFolds'])
    
    i = inputs.at[0,'Hyperparameters']
    if i == "DefaultValues":
        model = LogisticRegression(penalty = 'none', random_state = r_state)
    if i == "Customized":
        penalty = inputs.at[0,'lr_customized_penalty_cl']
        c = inputs.at[0,'lr_customized_c_cl']
        model = LogisticRegression(solver = 'liblinear', penalty = penalty, C = c, random_state = r_state)
        
    cv_results_ = cross_validate(model, x_train, y_train, scoring = ['accuracy','recall_weighted','precision_weighted','f1_weighted'], cv = folds)
    model.fit(x_train,y_train)
    train_accuracy = (cv_results_['test_accuracy'].mean()) 
    train_recall = (cv_results_['test_recall_weighted'].mean()) 
    train_precision = (cv_results_['test_precision_weighted'].mean()) 
    train_f1 = (cv_results_['test_f1_weighted'].mean()) 
    
    coef = model.coef_[0]
    print("Coef: ", coef)
    odds_ratio = np.exp(coef)
    print("Odds ratio: ", odds_ratio)
    feature_names = x_train.columns
    FI.f_importances(abs(coef), feature_names)
    
    j = inputs.at[0,'ModelOptions']
    if j == "None":
        train_accuracy, train_recall,train_precision, train_f1, test_accuracy, test_recall, test_precision, test_f1, test_time= pm.perfcl(inputs, model,train_accuracy,train_recall,train_precision, train_f1, x_train,y_train,x_test,y_test)
    if j == "Test with New Data":
        train_accuracy, train_recall,train_precision, train_f1, test_accuracy, test_recall, test_precision, test_f1 = pm.perfclnew(inputs,model,train_accuracy, train_recall,train_precision, train_f1)
    
    model_name = inputs.at[0,'Methods']
    parameters = model.get_params()
    
    return parameters, model_name, train_accuracy, train_recall,train_precision, train_f1, test_accuracy, test_recall, test_precision, test_f1, test_time, coef
    
def ordinal(inputs, x_train, y_train, x_test, y_test):
    import mord as m
    import numpy as np
    import perfmetric as pm
    from sklearn.model_selection import cross_validate
   
    y_train = np.array(y_train, dtype=np.float64)
    y_train = y_train.astype(np.int)
    y_test = np.array(y_test, dtype=np.float64)
    y_test = y_test.astype(np.int)
    
    folds = int(inputs.at[0,'CVFolds'])
    i = inputs.at[0,'Hyperparameters']
    if i == "DefaultValues":
        model = m.LogisticIT() #Default parameters: alpha=1.0, verbose=0, maxiter=10000
    if i == "Customized":
        alpha = inputs.at[0,'ord_customized_alpha_cl']
        model = m.LogisticIT(alpha = alpha) #Default parameters: alpha=1.0, verbose=0, maxiter=10000    
        
    cv_results_ = cross_validate(model, x_train, y_train, scoring = ['accuracy','recall_weighted','precision_weighted','f1_weighted'], cv = folds)
    model.fit(x_train,y_train)
    train_accuracy = (cv_results_['test_accuracy'].mean()) 
    train_recall = (cv_results_['test_recall_weighted'].mean()) 
    train_precision = (cv_results_['test_precision_weighted'].mean()) 
    train_f1 = (cv_results_['test_f1_weighted'].mean()) 
    
    coef = model.coef_
    print("Coef: ", coef)
    odds_ratio = np.exp(coef)
    print("Odds ratio: ", odds_ratio)
    
    j = inputs.at[0,'ModelOptions']
    if j == "None":
        train_accuracy, train_recall,train_precision, train_f1, test_accuracy, test_recall, test_precision, test_f1, test_time = pm.perfcl(inputs, model,train_accuracy,train_recall,train_precision, train_f1, x_train,y_train,x_test,y_test)
    if j == "Test with New Data":
        train_accuracy, train_recall,train_precision, train_f1, test_accuracy, test_recall, test_precision, test_f1 = pm.perfclnew(inputs,model,train_accuracy, train_recall,train_precision, train_f1)
    
    model_name = inputs.at[0,'Methods']
    parameters = model.get_params()
    
    return parameters, model_name, train_accuracy, train_recall,train_precision, train_f1, test_accuracy, test_recall, test_precision, test_f1, test_time, coef
  
def gb(inputs, x_train, y_train, x_test, y_test):
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import cross_validate
    import perfmetric as pm
    import pandas as pd
    
    seed = int(inputs.at[0,'RandomState'])
    if seed != 0:
        r_state = seed
    else:
        r_state = None

    folds = int(inputs.at[0,'CVFolds'])
    
    i = inputs.at[0,'Hyperparameters']
    if i == "DefaultValues":
        model = GradientBoostingClassifier(random_state = r_state)
    if i == "Customized":
        md = int(inputs.at[0,'gb_customized_md_cl'])
        ne = int(inputs.at[0,'gb_customized_ne_cl'])
        mss = int(inputs.at[0,'gb_customized_mss_cl'])
        lr = inputs.at[0,'gb_customized_lr_cl']
        model = GradientBoostingClassifier(max_depth = md, learning_rate = lr, min_samples_split = mss, n_estimators = ne, random_state = r_state)
        
    cv_results_ = cross_validate(model, x_train, y_train, scoring = ['accuracy','recall_weighted','precision_weighted','f1_weighted'], cv = folds)
    model.fit(x_train,y_train)
    train_accuracy = (cv_results_['test_accuracy'].mean()) 
    train_recall = (cv_results_['test_recall_weighted'].mean()) 
    train_precision = (cv_results_['test_precision_weighted'].mean()) 
    train_f1 = (cv_results_['test_f1_weighted'].mean()) 
    
    feat_importances = pd.Series(model.feature_importances_, index=x_train.columns)
    dependent_var = inputs.at[0,'Y']
    feat_importances.nlargest(20).plot(kind='barh', title = "Feature Importance for preciting " + dependent_var + " using Gradient Boosting Model")
    
      
    j = inputs.at[0,'ModelOptions']
    if j == "None":
        train_accuracy, train_recall,train_precision, train_f1, test_accuracy, test_recall, test_precision, test_f1, test_time = pm.perfcl(inputs, model,train_accuracy,train_recall,train_precision, train_f1, x_train,y_train,x_test,y_test)
    if j == "Test with New Data":
        train_accuracy, train_recall,train_precision, train_f1, test_accuracy, test_recall, test_precision, test_f1 = pm.perfclnew(inputs,model,train_accuracy, train_recall,train_precision, train_f1)
    
    model_name = inputs.at[0,'Methods']
    parameters = model.get_params()
    
    return parameters, model_name, train_accuracy, train_recall,train_precision, train_f1, test_accuracy, test_recall, test_precision, test_f1, test_time, feat_importances

def adacl(inputs, x_train, y_train, x_test, y_test):
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.model_selection import cross_validate
    import perfmetric as pm
    import pandas as pd
    
    seed = int(inputs.at[0,'RandomState'])
    if seed != 0:
        r_state = seed
    else:
        r_state = None

    folds = int(inputs.at[0,'CVFolds'])

    i = inputs.at[0,'Hyperparameters']
    if i == "DefaultValues":
        model = AdaBoostClassifier(random_state = r_state)
    if i == "Customized":
        lr = inputs.at[0,'ada_customized_lr_cl']
        ne = int(inputs.at[0,'ada_customized_ne_cl'])
        algorithm = inputs.at[0,'ada_customized_algorithm_cl']
        model = AdaBoostClassifier(learning_rate = lr, n_estimators = ne, algorithm = algorithm, random_state = r_state)
        
    cv_results_ = cross_validate(model, x_train, y_train, scoring = ['accuracy','recall_weighted','precision_weighted','f1_weighted'], cv = folds)
    model.fit(x_train,y_train)
    train_accuracy = (cv_results_['test_accuracy'].mean()) 
    train_recall = (cv_results_['test_recall_weighted'].mean()) 
    train_precision = (cv_results_['test_precision_weighted'].mean()) 
    train_f1 = (cv_results_['test_f1_weighted'].mean()) 
    
    feat_importances = pd.Series(model.feature_importances_, index=x_train.columns)
    feat_importances.nlargest(20).plot(kind='barh')

    j = inputs.at[0,'ModelOptions']
    if j == "None":
        train_accuracy, train_recall,train_precision, train_f1, test_accuracy, test_recall, test_precision, test_f1, test_time = pm.perfcl(inputs,model,train_accuracy,train_recall,train_precision, train_f1, x_train,y_train,x_test,y_test)
    if j == "Test with New Data":
        train_accuracy, train_recall,train_precision, train_f1, test_accuracy, test_recall, test_precision, test_f1 = pm.perfclnew(inputs,model,train_accuracy, train_recall,train_precision, train_f1)
    
    model_name = inputs.at[0,'Methods']
    parameters = model.get_params()
    
    return parameters, model_name, train_accuracy, train_recall,train_precision, train_f1, test_accuracy, test_recall, test_precision, test_f1,test_time, feat_importances

def xgbcl(inputs, x_train, y_train, x_test, y_test):
    import xgboost as xgb
    from sklearn.model_selection import cross_validate
    import perfmetric as pm
    import pandas as pd
    
    folds = int(inputs.at[0,'CVFolds'])
    
    i = inputs.at[0,'Hyperparameters']
    if i == "DefaultValues":
        model = xgb.XGBClassifier()
    if i == "Customized":
        md = int(inputs.at[0,'xgb_customized_md_cl'])
        lr = inputs.at[0,'xgb_customized_lr_cl']
        ne = int(inputs.at[0,'xgb_customized_ne_cl'])
        model = xgb.XGBClassifier(max_depth = md, learning_rate= lr, n_estimators = ne)
        
    cv_results_ = cross_validate(model, x_train, y_train, scoring = ['accuracy','recall_weighted','precision_weighted','f1_weighted'], cv = folds)
    model.fit(x_train, y_train)
    train_accuracy = (cv_results_['test_accuracy'].mean()) 
    train_recall = (cv_results_['test_recall_weighted'].mean()) 
    train_precision = (cv_results_['test_precision_weighted'].mean()) 
    train_f1 = (cv_results_['test_f1_weighted'].mean()) 
    
    feat_importances = pd.Series(model.feature_importances_, index=x_train.columns)
    feat_importances.nlargest(20).plot(kind='barh')

    j = inputs.at[0,'ModelOptions']
    if j == "None":
        train_accuracy, train_recall,train_precision, train_f1, test_accuracy, test_recall, test_precision, test_f1, test_time = pm.perfcl(inputs,model,train_accuracy,train_recall,train_precision, train_f1, x_train,y_train,x_test,y_test)
    if j == "Test with New Data":
        train_accuracy, train_recall,train_precision, train_f1, test_accuracy, test_recall, test_precision, test_f1 = pm.perfclnew(inputs,model,train_accuracy, train_recall,train_precision, train_f1)
    
    model_name = inputs.at[0,'Methods']
    parameters = model.get_params()
    
    return parameters, model_name, train_accuracy, train_recall,train_precision, train_f1, test_accuracy, test_recall, test_precision, test_f1, test_time, feat_importances

