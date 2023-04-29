#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 31 17:00:16 2020

@author: HarounJr
"""
def gradientBoosting(inputs, x_train, y_train, x_test, y_test):
    from sklearn.ensemble import GradientBoostingRegressor
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
        model = GradientBoostingRegressor(random_state = r_state)
    if i == "Customized":
        md = int(inputs.at[0,'gb_customized_md'])
        ne = int(inputs.at[0,'gb_customized_ne'])
        mss = int(inputs.at[0,'gb_customized_mss'])
        lr = inputs.at[0,'gb_customized_lr']
        model = GradientBoostingRegressor(max_depth = md, learning_rate = lr, min_samples_split = mss, n_estimators = ne, random_state = r_state)
     
    cv_results_ = cross_validate(model, x_train, y_train, scoring = ['neg_mean_squared_error','neg_mean_absolute_error','neg_root_mean_squared_error','r2'], cv = folds)
    model.fit(x_train,y_train)
    
    train_mse = ((cv_results_['test_neg_mean_squared_error'].mean()) * -1)
    train_mae = ((cv_results_['test_neg_mean_absolute_error'].mean()) * -1)
    train_rmse = ((cv_results_['test_neg_root_mean_squared_error'].mean()) * -1)
    train_r2 = (cv_results_['test_r2'].mean()) 
   
    feat_importances = pd.Series(model.feature_importances_, index=x_train.columns)
    dependent_var = inputs.at[0,'Y']
    feat_importances.nlargest(20).plot(kind='barh', title = "Feature Importance for preciting " + dependent_var + " using Gradient Boosting Model")
    
      
    j = inputs.at[0,'ModelOptions']
    if j == "None":
         train_mse, train_mae, train_rmse, train_r2, test_mse, test_mae, test_rmse,  test_r2, test_time = pm.perfreg(inputs,model,train_mse, train_mae, train_rmse, train_r2, x_train,y_train,x_test,y_test)
    if j == "Test with New Data":
         train_mse, train_mae, train_rmse,  train_r2, test_mse, test_mae, test_rmse,  test_r2 = pm.perfregnew(inputs,model,train_mse, train_mae, train_rmse, train_r2)
    
    model_name = inputs.at[0,'Methods']
    parameters = model.get_params()
    
    return parameters, model_name, train_mse, train_mae, train_rmse,  train_r2, test_mse, test_mae, test_rmse, test_r2, test_time, feat_importances
    
def adaboost(inputs, x_train, y_train, x_test, y_test):
    from sklearn.ensemble import AdaBoostRegressor
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
        model = AdaBoostRegressor(random_state = r_state)
    if i == "Customized":
        lr = inputs.at[0,'ada_customized_lr']
        ne = int(inputs.at[0,'ada_customized_ne'])
        loss = str(inputs.at[0,'ada_customized_loss'])
        model = AdaBoostRegressor(learning_rate = lr, n_estimators = ne, loss = loss, random_state = r_state)
           
    cv_results_ = cross_validate(model, x_train, y_train, scoring = ['neg_mean_squared_error','neg_mean_absolute_error','neg_root_mean_squared_error','r2'], cv = folds)
    model.fit(x_train,y_train)
    
    train_mse = ((cv_results_['test_neg_mean_squared_error'].mean()) * -1)
    train_mae = ((cv_results_['test_neg_mean_absolute_error'].mean()) * -1)
    train_rmse = ((cv_results_['test_neg_root_mean_squared_error'].mean()) * -1)
    train_r2 = (cv_results_['test_r2'].mean()) 
    
    feat_importances = pd.Series(model.feature_importances_, index=x_train.columns)
    dependent_var = inputs.at[0,'Y']
    feat_importances.nlargest(20).plot(kind='barh', title = "Feature Importance for preciting " + dependent_var + " using Gradient Boosting Model")
    
      
    j = inputs.at[0,'ModelOptions']
    if j == "None":
         train_mse, train_mae, train_rmse, train_r2, test_mse, test_mae, test_rmse,  test_r2, test_time = pm.perfreg(inputs,model,train_mse, train_mae, train_rmse, train_r2, x_train,y_train,x_test,y_test)
    if j == "Test with New Data":
         train_mse, train_mae, train_rmse,  train_r2, test_mse, test_mae, test_rmse,  test_r2 = pm.perfregnew(inputs,model,train_mse, train_mae, train_rmse, train_r2)
    
    model_name = inputs.at[0,'Methods']
    parameters = model.get_params()
    
    return parameters, model_name, train_mse, train_mae, train_rmse,  train_r2, test_mse, test_mae, test_rmse,  test_r2, test_time, feat_importances 
 
def xgboost(inputs, x_train, y_train, x_test, y_test):
    import xgboost as xgb
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
           model = xgb.XGBRegressor(random_state=r_state)
    if i == "Customized":
        md = int(inputs.at[0,'xgb_customized_md'])
        lr = inputs.at[0,'xgb_customized_lr']
        ne = int(inputs.at[0,'xgb_customized_ne'])
        model = xgb.XGBRegressor(max_depth = md, learning_rate= lr, n_estimators = ne)
 
    cv_results_ = cross_validate(model, x_train, y_train, scoring = ['neg_mean_squared_error','neg_mean_absolute_error','neg_root_mean_squared_error','r2'], cv = folds)
    model.fit(x_train,y_train)
    
    train_mse = ((cv_results_['test_neg_mean_squared_error'].mean()) * -1)
    train_mae = ((cv_results_['test_neg_mean_absolute_error'].mean()) * -1)
    train_rmse = ((cv_results_['test_neg_root_mean_squared_error'].mean()) * -1)
    train_r2 = (cv_results_['test_r2'].mean()) 
    
    feat_importances = pd.Series(model.feature_importances_, index=x_train.columns)
    feat_importances.nlargest(20).plot(kind='barh')

    j = inputs.at[0,'ModelOptions']
    if j == "None":
         train_mse, train_mae, train_rmse, train_r2, test_mse, test_mae, test_rmse,  test_r2, test_time = pm.perfreg(inputs,model,train_mse, train_mae, train_rmse, train_r2, x_train,y_train,x_test,y_test)
    if j == "Test with New Data":
         train_mse, train_mae, train_rmse,  train_r2, test_mse, test_mae, test_rmse,  test_r2 = pm.perfregnew(inputs,model,train_mse, train_mae, train_rmse, train_r2)
    
    model_name = inputs.at[0,'Methods']
    parameters = model.get_params()
    
    return parameters, model_name, train_mse, train_mae, train_rmse,  train_r2, test_mse, test_mae, test_rmse,  test_r2, test_time, feat_importances
 
def rf(inputs, x_train, y_train, x_test, y_test):
    from sklearn.ensemble import RandomForestRegressor
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
        model = RandomForestRegressor(random_state = r_state)
    if i == "Customized":
        ne = int(inputs.at[0,'rf_customized_ne'])
        msl = int(inputs.at[0,'rf_customized_msl'])
        mf = inputs.at[0,'rf_customized_mf']
        model = RandomForestRegressor(max_features = mf, min_samples_leaf = msl, n_estimators = ne, random_state = r_state)
     
    cv_results_ = cross_validate(model, x_train, y_train, scoring = ['neg_mean_squared_error','neg_mean_absolute_error','neg_root_mean_squared_error','r2'], cv = folds)
    model.fit(x_train,y_train)
    
    train_mse = ((cv_results_['test_neg_mean_squared_error'].mean()) * -1)
    train_mae = ((cv_results_['test_neg_mean_absolute_error'].mean()) * -1)
    train_rmse = ((cv_results_['test_neg_root_mean_squared_error'].mean()) * -1)
    train_r2 = (cv_results_['test_r2'].mean()) 
   
    feat_importances = pd.Series(model.feature_importances_, index=x_train.columns)
    dependent_var = inputs.at[0,'Y']
    feat_importances.nlargest(20).plot(kind='barh', title = "Feature Importance for preciting " + dependent_var + " using Gradient Boosting Model")
    
      
    j = inputs.at[0,'ModelOptions']
    if j == "None":
         train_mse, train_mae, train_rmse, train_r2, test_mse, test_mae, test_rmse,  test_r2, test_time = pm.perfreg(inputs,model,train_mse, train_mae, train_rmse, train_r2, x_train,y_train,x_test,y_test)
    if j == "Test with New Data":
         train_mse, train_mae, train_rmse,  train_r2, test_mse, test_mae, test_rmse,  test_r2 = pm.perfregnew(inputs,model,train_mse, train_mae, train_rmse, train_r2)
    
    model_name = inputs.at[0,'Methods']
    parameters = model.get_params()
    
    return parameters, model_name, train_mse, train_mae, train_rmse,  train_r2, test_mse, test_mae, test_rmse,  test_r2, test_time, feat_importances
     
def lr(inputs, x_train, y_train, x_test, y_test):
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import cross_validate
    import perfmetric as pm
        
    folds = int(inputs.at[0,'CVFolds'])
    
    i = inputs.at[0,'Hyperparameters']
    if i == "DefaultValues":
        model = LinearRegression()
    if i == "Customized":
       print("Option doesn't exit for this model yet")
       
    cv_results_ = cross_validate(model, x_train, y_train, scoring = ['neg_mean_squared_error','neg_mean_absolute_error','neg_root_mean_squared_error','r2'], cv = folds)
    model.fit(x_train,y_train)
    
    train_mse = ((cv_results_['test_neg_mean_squared_error'].mean()) * -1)
    train_mae = ((cv_results_['test_neg_mean_absolute_error'].mean()) * -1)
    train_rmse = ((cv_results_['test_neg_root_mean_squared_error'].mean()) * -1)
    train_r2 = (cv_results_['test_r2'].mean()) 
   
    coef = model.coef_[0]
    print("Coef: ", coef)

      
    j = inputs.at[0,'ModelOptions']
    if j == "None":
         train_mse, train_mae, train_rmse, train_r2, test_mse, test_mae, test_rmse,  test_r2, test_time = pm.perfreg(inputs, model,train_mse, train_mae, train_rmse, train_r2, x_train,y_train,x_test,y_test)
    if j == "Test with New Data":
         train_mse, train_mae, train_rmse,  train_r2, test_mse, test_mae, test_rmse,  test_r2 = pm.perfregnew(inputs,model,train_mse, train_mae, train_rmse, train_r2)
    
    model_name = inputs.at[0,'Methods']
    parameters = model.get_params()
    
    return parameters, model_name, train_mse, train_mae, train_rmse,  train_r2, test_mse, test_mae, test_rmse,  test_r2, test_time, coef
     


