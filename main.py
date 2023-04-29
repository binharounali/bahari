#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 12:08:49 2020

@author: HarounJr
"""

#Defining the logic of the app
def main(inputs,result_file):
    import optimization as op
    import regression as reg
    import classification as cl
    import preprocess as pp
    import pandas as pd
    import files as files
    import time


    #File and variables
    file = inputs.at[0,'Data File']
    df = pd.read_excel (file)
    xs = inputs['Xs'].values[0].split(",")
    ys = inputs.at[0,'Y']
    x = pd.DataFrame(df, columns= xs)
    y = pd.DataFrame(df, columns= [ys])
    
    #Data Imputation
    x,y = pp.impute(inputs,x,y)
    
    #Feature Scaling
    featurescale = inputs.at[0,'FeatureScale']
    if featurescale == "Yes":
        x,y = pp.scale(inputs,x,y)
    
    #Data Splitting
    x_train, x_test, y_train, y_test = pp.datasplit(inputs,x,y)
    
    i = inputs.at[0,'MLCategory']
    if i == "Classification":
        mod_i = inputs.at[0,'Methods']
        
        if mod_i == "RandomForest":
            i = inputs.at[0,'Hyperparameters']
            if i == "DefaultValues":
                start = time.time()
                parameters, model_name, train_accuracy, train_recall, train_precision, train_f1,test_accuracy, test_recall, test_precision, test_f1, test_time, fi = cl.rf(inputs, x_train, y_train, x_test, y_test) 
                end = time.time()
                time = end-start
                files.updatefilecl(inputs,result_file,model_name,"None", time, "None", test_time, train_accuracy,train_recall, train_precision, train_f1, test_accuracy, test_recall, test_precision, test_f1, parameters, fi)
            elif i == "Tune":
                i = inputs.at[0,'Tune']
                if i == "Grid Search":
                    model_name, model, parameters = op.clmodel(inputs,mod_i)
                    start = time.time()
                    opt_method, train_accuracy, train_recall, train_precision, train_f1, test_accuracy, test_recall, test_precision, test_f1, best_para, tune_time, test_time, fi = op.gridcl(model_name, inputs,model,parameters,x_train,y_train,x_test,y_test)
                    end = time.time()
                    time = end-start
                    files.updatefilecl(inputs,result_file,model_name, opt_method,time,tune_time,test_time, train_accuracy, train_recall, train_precision, train_f1, test_accuracy, test_recall, test_precision, test_f1, best_para, fi)
                elif i == "Random Search":
                    model_name, model, parameters = op.clmodel(inputs,mod_i)
                    start = time.time()
                    opt_method, train_accuracy,train_recall, train_precision, train_f1, test_accuracy, test_recall, test_precision, test_f1, best_para, tune_time, test_time, fi = op.randomcl(model_name, inputs,model,parameters,x_train,y_train,x_test,y_test)
                    end = time.time()
                    time = end-start
                    files.updatefilecl(inputs,result_file,model_name, opt_method,time,tune_time,test_time, train_accuracy, train_recall, train_precision, train_f1, test_accuracy, test_recall, test_precision, test_f1, best_para, fi)
                elif i == "Bayes Opt":
                    start = time.time()
                    opt_method,model_name, train_mse, train_mae, train_rmse, train_r2, test_mse, test_mae, test_rmse,  test_r2, best_para, tuning_time,test_time, fi = op.bayescl(inputs,x_train,y_train,x_test,y_test)
                    end = time.time()
                    time = end-start
                    files.updatefilereg(inputs,result_file,model_name, opt_method,time, tuning_time,test_time, train_mse, train_mae, train_rmse, train_r2, test_mse, test_mae, test_rmse,  test_r2, best_para, fi)

            elif i == "Customized":
                start = time.time()
                parameters, model_name, train_accuracy, train_recall, train_precision, train_f1, test_accuracy, test_recall, test_precision, test_f1,test_time, fi = cl.rf(inputs, x_train, y_train, x_test, y_test) 
                end = time.time()
                time = end-start
                files.updatefilecl(inputs,result_file,model_name,"None", time, "None",test_time, train_accuracy,train_recall, train_precision, train_f1, test_accuracy, test_recall, test_precision, test_f1, parameters, fi)

        if mod_i == "SVM":
            i = inputs.at[0,'Hyperparameters']
            if i == "DefaultValues":
                start = time.time()
                parameters, model_name, train_accuracy, train_recall, train_precision, train_f1, test_accuracy , test_recall, test_precision, test_f1,test_time, fi = cl.svm(inputs, x_train, y_train, x_test, y_test) 
                end = time.time()
                time = end-start
                files.updatefilecl(inputs,result_file,model_name,"None", time, "None", test_time, train_accuracy,train_recall, train_precision, train_f1, test_accuracy, test_recall, test_precision, test_f1, parameters, fi)
            elif i == "Tune":
                i = inputs.at[0,'Tune']
                if i == "Grid Search":
                    model_name, model, parameters = op.clmodel(inputs,mod_i)
                    start = time.time()
                    opt_method, train_accuracy, train_recall, train_precision, train_f1, test_accuracy, test_recall, test_precision, test_f1, best_para, tune_time, test_time, fi = op.gridcl(model_name, inputs,model,parameters,x_train,y_train,x_test,y_test)
                    end = time.time()
                    time = end-start
                    files.updatefilecl(inputs,result_file,model_name, opt_method,time,tune_time,test_time, train_accuracy, train_recall, train_precision, train_f1, test_accuracy, test_recall, test_precision, test_f1, best_para, fi)
                elif i == "Random Search":
                    model_name, model, parameters = op.clmodel(inputs,mod_i)
                    start = time.time()
                    opt_method, train_accuracy,train_recall, train_precision, train_f1, test_accuracy, test_recall, test_precision, test_f1, best_para, tune_time, test_time, fi = op.randomcl(model_name, inputs,model,parameters,x_train,y_train,x_test,y_test)
                    end = time.time()
                    time = end-start
                    files.updatefilecl(inputs,result_file,model_name, opt_method,time,tune_time,test_time,train_accuracy, train_recall, train_precision, train_f1, test_accuracy, test_recall, test_precision, test_f1, best_para, fi)
                elif i == "Bayes Opt":
                    start = time.time()
                    opt_method,model_name, train_mse, train_mae, train_rmse, train_r2, test_mse, test_mae, test_rmse,  test_r2, best_para, tuning_time,test_time, fi = op.bayescl(inputs,x_train,y_train,x_test,y_test)
                    end = time.time()
                    time = end-start
                    files.updatefilereg(inputs,result_file,model_name, opt_method,time, tuning_time,test_time, train_mse, train_mae, train_rmse, train_r2, test_mse, test_mae, test_rmse,  test_r2, best_para, fi)

            elif i == "Customized":
                start = time.time()
                parameters, model_name, train_accuracy, train_recall, train_precision, train_f1, test_accuracy , test_recall, test_precision, test_f1, test_time, fi = cl.svm(inputs, x_train, y_train, x_test, y_test) 
                end = time.time()
                time = end-start
                files.updatefilecl(inputs,result_file,model_name,"None", time,"None", test_time, train_accuracy,train_recall, train_precision, train_f1, test_accuracy, test_recall, test_precision, test_f1, parameters, fi)

        if mod_i == "Default_LR":
            i = inputs.at[0,'Hyperparameters']
            if i == "DefaultValues":
                start = time.time()
                parameters, model_name, train_accuracy, train_recall, train_precision, train_f1, test_accuracy, test_recall, test_precision, test_f1, test_time, fi = cl.logisticReg(inputs, x_train, y_train, x_test, y_test) 
                end = time.time()
                time = end-start
                files.updatefilecl(inputs,result_file,model_name,"None", time, "None", test_time, train_accuracy,train_recall, train_precision, train_f1, test_accuracy, test_recall, test_precision, test_f1, parameters, fi)
            elif i == "Tune":
                i = inputs.at[0,'Tune']
                if i == "Grid Search":
                    model_name, model, parameters = op.clmodel(inputs,mod_i)
                    start = time.time()
                    opt_method, train_accuracy, train_recall, train_precision, train_f1, test_accuracy, test_recall, test_precision, test_f1, best_para, tune_time, test_time, fi = op.gridcl(model_name, inputs,model,parameters,x_train,y_train,x_test,y_test)
                    end = time.time()
                    time = end-start
                    files.updatefilecl(inputs,result_file,model_name, opt_method,time,tune_time,test_time, train_accuracy, train_recall, train_precision, train_f1, test_accuracy, test_recall, test_precision, test_f1, best_para, fi)
                elif i == "Bayes Opt":
                    start = time.time()
                    opt_method,model_name, train_mse, train_mae, train_rmse, train_r2, test_mse, test_mae, test_rmse,  test_r2, best_para, tuning_time,test_time, fi = op.bayescl(inputs,x_train,y_train,x_test,y_test)
                    end = time.time()
                    time = end-start
                    files.updatefilereg(inputs,result_file,model_name, opt_method,time, tuning_time,test_time, train_mse, train_mae, train_rmse, train_r2, test_mse, test_mae, test_rmse,  test_r2, best_para, fi)
                elif i == "Random Search":
                    model_name, model, parameters = op.clmodel(inputs,mod_i)
                    start = time.time()
                    opt_method, train_accuracy,train_recall, train_precision, train_f1, test_accuracy, test_recall, test_precision, test_f1, best_para, tune_time,test_time, fi = op.randomcl(model_name, inputs,model,parameters,x_train,y_train,x_test,y_test)
                    end = time.time()
                    time = end-start
                    files.updatefilecl(inputs,result_file,model_name, opt_method,time,tune_time,test_time,train_accuracy, train_recall, train_precision, train_f1, test_accuracy, test_recall, test_precision, test_f1, best_para, fi)
            elif i == "Customized":
                start = time.time()
                parameters, model_name, train_accuracy, train_recall, train_precision, train_f1, test_accuracy, test_recall, test_precision, test_f1,test_time, fi = cl.logisticReg(inputs, x_train, y_train, x_test, y_test) 
                end = time.time()
                time = end-start
                files.updatefilecl(inputs,result_file,model_name,"None", time, "None",test_time, train_accuracy,train_recall, train_precision, train_f1, test_accuracy, test_recall, test_precision, test_f1, parameters, fi)

        if mod_i == "Ordinal_LR":
            i = inputs.at[0,'Hyperparameters']
            if i == "DefaultValues":
                start = time.time()
                parameters, model_name, train_accuracy, train_recall, train_precision, train_f1, test_accuracy, test_recall, test_precision, test_f1, test_time, fi = cl.ordinal(inputs, x_train, y_train, x_test, y_test) 
                end = time.time()
                time = end-start
                files.updatefilecl(inputs,result_file,model_name,"None", time, "None", test_time,train_accuracy,train_recall, train_precision, train_f1, test_accuracy, test_recall, test_precision, test_f1, parameters, fi)
            elif i == "Tune":
                i = inputs.at[0,'Tune']
                if i == "Grid Search":
                    model_name, model, parameters = op.clmodel(inputs,mod_i)
                    start = time.time()
                    opt_method, train_accuracy, train_recall, train_precision, train_f1, test_accuracy, test_recall, test_precision, test_f1, best_para, tune_time, test_time, fi = op.gridcl(model_name, inputs,model,parameters,x_train,y_train,x_test,y_test)
                    end = time.time()
                    time = end-start
                    files.updatefilecl(inputs,result_file,model_name, opt_method,time,tune_time,test_time, train_accuracy, train_recall, train_precision, train_f1, test_accuracy, test_recall, test_precision, test_f1, best_para, fi)
                elif i == "Random Search":
                    model_name, model, parameters = op.clmodel(inputs,mod_i)
                    start = time.time()
                    opt_method, train_accuracy,train_recall, train_precision, train_f1, test_accuracy, test_recall, test_precision, test_f1, best_para, tune_time, test_time,fi = op.randomcl(model_name, inputs,model,parameters,x_train,y_train,x_test,y_test)
                    end = time.time()
                    time = end-start
                    files.updatefilecl(inputs,result_file,model_name, opt_method,time,tune_time,test_time,train_accuracy, train_recall, train_precision, train_f1, test_accuracy, test_recall, test_precision, test_f1, best_para,fi)
                elif i == "Bayes Opt":
                    start = time.time()
                    opt_method,model_name, train_mse, train_mae, train_rmse, train_r2, test_mse, test_mae, test_rmse,  test_r2, best_para, tuning_time,test_time, fi = op.bayescl(inputs,x_train,y_train,x_test,y_test)
                    end = time.time()
                    time = end-start
                    files.updatefilereg(inputs,result_file,model_name, opt_method,time, tuning_time,test_time, train_mse, train_mae, train_rmse, train_r2, test_mse, test_mae, test_rmse,  test_r2, best_para, fi)
            elif i == "Customized":
                start = time.time()
                parameters, model_name, train_accuracy, train_recall, train_precision, train_f1, test_accuracy, test_recall, test_precision, test_f1, test_time, fi = cl.ordinal(inputs, x_train, y_train, x_test, y_test) 
                end = time.time()
                time = end-start
                files.updatefilecl(inputs,result_file,model_name,"None", time, "None", test_time, train_accuracy,train_recall, train_precision, train_f1, test_accuracy, test_recall, test_precision, test_f1, parameters, fi)

        if mod_i == "GradientBoosting":
            i = inputs.at[0,'Hyperparameters']
            if i == "DefaultValues":
                start = time.time()
                parameters, model_name, train_accuracy, train_recall, train_precision, train_f1,test_accuracy, test_recall, test_precision, test_f1, test_time,fi = cl.gb(inputs, x_train, y_train, x_test, y_test) 
                end = time.time()
                time = end-start
                files.updatefilecl(inputs,result_file,model_name,"None", time, "None", test_time,train_accuracy,train_recall, train_precision, train_f1, test_accuracy, test_recall, test_precision, test_f1, parameters, fi)
            elif i == "Tune":
                i = inputs.at[0,'Tune']
                if i == "Grid Search":
                    model_name, model, parameters = op.clmodel(inputs,mod_i)
                    start = time.time()
                    opt_method, train_accuracy, train_recall, train_precision, train_f1, test_accuracy, test_recall, test_precision, test_f1, best_para, tune_time, test_time, fi = op.gridcl(model_name, inputs,model,parameters,x_train,y_train,x_test,y_test)
                    end = time.time()
                    time = end-start
                    files.updatefilecl(inputs,result_file,model_name, opt_method,time,tune_time,test_time,train_accuracy, train_recall, train_precision, train_f1, test_accuracy, test_recall, test_precision, test_f1, best_para, fi)
                elif i == "Random Search":
                    model_name, model, parameters = op.clmodel(inputs,mod_i)
                    start = time.time()
                    opt_method, train_accuracy,train_recall, train_precision, train_f1, test_accuracy, test_recall, test_precision, test_f1, best_para, tune_time,test_time,fi = op.randomcl(model_name, inputs,model,parameters,x_train,y_train,x_test,y_test)
                    end = time.time()
                    time = end-start
                    files.updatefilecl(inputs,result_file,model_name, opt_method,time,tune_time,test_time,train_accuracy, train_recall, train_precision, train_f1, test_accuracy, test_recall, test_precision, test_f1, best_para,fi)
                elif i == "Bayes Opt":
                    start = time.time()
                    opt_method,model_name, train_mse, train_mae, train_rmse, train_r2, test_mse, test_mae, test_rmse,  test_r2, best_para, tuning_time,test_time, fi = op.bayescl(inputs,x_train,y_train,x_test,y_test)
                    end = time.time()
                    time = end-start
                    files.updatefilereg(inputs,result_file,model_name, opt_method,time, tuning_time,test_time, train_mse, train_mae, train_rmse, train_r2, test_mse, test_mae, test_rmse,  test_r2, best_para, fi)
            elif i == "Customized":
                start = time.time()
                parameters, model_name, train_accuracy, train_recall, train_precision, train_f1, test_accuracy, test_recall, test_precision, test_f1, test_time, fi = cl.gb(inputs, x_train, y_train, x_test, y_test) 
                end = time.time()
                time = end-start
                files.updatefilecl(inputs,result_file,model_name,"None", time, "None", test_time,train_accuracy,train_recall, train_precision, train_f1, test_accuracy, test_recall, test_precision, test_f1, parameters, fi)

        if mod_i == "AdaBoost":
            i = inputs.at[0,'Hyperparameters']
            if i == "DefaultValues":
                start = time.time()
                parameters, model_name, train_accuracy, train_recall, train_precision, train_f1, test_accuracy, test_recall, test_precision, test_f1, test_time,fi = cl.adacl(inputs, x_train, y_train, x_test, y_test) 
                end = time.time()
                time = end-start
                files.updatefilecl(inputs,result_file,model_name,"None", time, "None", test_time, train_accuracy,train_recall, train_precision, train_f1, test_accuracy, test_recall, test_precision, test_f1, parameters, fi)
            elif i == "Tune":
                i = inputs.at[0,'Tune']
                if i == "Grid Search":
                    model_name, model, parameters = op.clmodel(inputs,mod_i)
                    start = time.time()
                    opt_method, train_accuracy, train_recall, train_precision, train_f1, test_accuracy, test_recall, test_precision, test_f1, best_para, tune_time, test_time, fi = op.gridcl(model_name, inputs,model,parameters,x_train,y_train,x_test,y_test)
                    end = time.time()
                    time = end-start
                    files.updatefilecl(inputs,result_file,model_name, opt_method,time,tune_time,test_time,train_accuracy, train_recall, train_precision, train_f1, test_accuracy, test_recall, test_precision, test_f1, best_para, fi)
                elif i == "Random Search":
                    model_name, model, parameters = op.clmodel(inputs,mod_i)
                    start = time.time()
                    opt_method, train_accuracy,train_recall, train_precision, train_f1, test_accuracy, test_recall, test_precision, test_f1, best_para, tune_time,test_time, fi = op.randomcl(model_name, inputs,model,parameters,x_train,y_train,x_test,y_test)
                    end = time.time()
                    time = end-start
                    files.updatefilecl(inputs,result_file,model_name, opt_method,time,tune_time,test_time, train_accuracy, train_recall, train_precision, train_f1, test_accuracy, test_recall, test_precision, test_f1, best_para,fi)
                elif i == "Bayes Opt":
                    start = time.time()
                    opt_method,model_name, train_mse, train_mae, train_rmse, train_r2, test_mse, test_mae, test_rmse,  test_r2, best_para, tuning_time,test_time, fi = op.bayescl(inputs,x_train,y_train,x_test,y_test)
                    end = time.time()
                    time = end-start
                    files.updatefilereg(inputs,result_file,model_name, opt_method,time, tuning_time,test_time, train_mse, train_mae, train_rmse, train_r2, test_mse, test_mae, test_rmse,  test_r2, best_para, fi)
            elif i == "Customized":
                start = time.time()
                parameters, model_name, train_accuracy, train_recall, train_precision, train_f1, test_accuracy, test_recall, test_precision, test_f1, test_time, fi = cl.adacl(inputs, x_train, y_train, x_test, y_test) 
                end = time.time()
                time = end-start
                files.updatefilecl(inputs,result_file,model_name,"None", time, "None", test_time, train_accuracy,train_recall, train_precision, train_f1, test_accuracy, test_recall, test_precision, test_f1, parameters, fi)

        if mod_i == "XgBoost":
            i = inputs.at[0,'Hyperparameters']
            if i == "DefaultValues":
                start = time.time()
                parameters, model_name, train_accuracy, train_recall, train_precision, train_f1, test_accuracy, test_recall, test_precision, test_f1, test_time, fi = cl.xgbcl(inputs, x_train, y_train, x_test, y_test) 
                end = time.time()
                time = end-start
                files.updatefilecl(inputs,result_file,model_name,"None", time, "None", test_time, train_accuracy,train_recall, train_precision, train_f1, test_accuracy, test_recall, test_precision, test_f1, parameters, fi)
            elif i == "Tune":
                i = inputs.at[0,'Tune']
                if i == "Grid Search":
                    model_name, model, parameters = op.clmodel(inputs,mod_i)
                    start = time.time()
                    opt_method, train_accuracy, train_recall, train_precision, train_f1, test_accuracy, test_recall, test_precision, test_f1, best_para, tune_time, test_time, fi = op.gridcl(model_name, inputs,model,parameters,x_train,y_train,x_test,y_test)
                    end = time.time()
                    time = end-start
                    files.updatefilecl(inputs,result_file,model_name, opt_method,time,tune_time,test_time,train_accuracy, train_recall, train_precision, train_f1, test_accuracy, test_recall, test_precision, test_f1, best_para, fi)
                elif i == "Random Search":
                    model_name, model, parameters = op.clmodel(inputs,mod_i)
                    start = time.time()
                    opt_method, train_accuracy,train_recall, train_precision, train_f1, test_accuracy, test_recall, test_precision, test_f1, best_para, tune_time, test_time, fi = op.randomcl(model_name, inputs,model,parameters,x_train,y_train,x_test,y_test)
                    end = time.time()
                    time = end-start
                    files.updatefilecl(inputs,result_file,model_name, opt_method,time,tune_time,test_time, train_accuracy, train_recall, train_precision, train_f1, test_accuracy, test_recall, test_precision, test_f1, best_para,fi)
                elif i == "Bayes Opt":
                    start = time.time()
                    opt_method,model_name, train_mse, train_mae, train_rmse, train_r2, test_mse, test_mae, test_rmse,  test_r2, best_para, tuning_time,test_time, fi = op.bayescl(inputs,x_train,y_train,x_test,y_test)
                    end = time.time()
                    time = end-start
                    files.updatefilereg(inputs,result_file,model_name, opt_method,time, tuning_time,test_time, train_mse, train_mae, train_rmse, train_r2, test_mse, test_mae, test_rmse,  test_r2, best_para, fi)
            elif i == "Customized":
                start = time.time()
                parameters, model_name, train_accuracy, train_recall, train_precision, train_f1, test_accuracy, test_recall, test_precision, test_f1, test_time, fi = cl.xgbcl(inputs, x_train, y_train, x_test, y_test) 
                end = time.time()
                time = end-start
                files.updatefilecl(inputs,result_file,model_name,"None", time, "None", test_time, train_accuracy,train_recall, train_precision, train_f1, test_accuracy, test_recall, test_precision, test_f1, parameters,fi)

    elif i == "Regression":  
        mod_i = inputs.at[0,'Methods']
        
        if mod_i == "GradientBoosting":
            i = inputs.at[0,'Hyperparameters']
            if i == "DefaultValues":
                start = time.time()
                parameters, model_name, train_mse, train_mae, train_rmse, train_r2, test_mse, test_mae, test_rmse, test_r2, test_time, fi = reg.gradientBoosting(inputs, x_train, y_train, x_test, y_test) 
                end = time.time()
                time = end-start
                files.updatefilereg(inputs,result_file,model_name,"None", time,"None", test_time, train_mse, train_mae, train_rmse, train_r2, test_mse, test_mae, test_rmse,  test_r2, parameters, fi)
            elif i == "Tune":
                i = inputs.at[0,'Tune']
                if i == "Grid Search":
                    start = time.time()
                    model_name, model, parameters = op.regmodel(inputs,mod_i)
                    opt_method, train_mse, train_mae, train_rmse, train_r2, test_mse, test_mae, test_rmse, test_r2, best_para, tuning_time, test_time, fi = op.grid(model_name, inputs,model,parameters,x_train,y_train,x_test,y_test)
                    end = time.time()
                    time = end-start
                    files.updatefilereg(inputs,result_file,model_name, opt_method,time, tuning_time, test_time, train_mse, train_mae, train_rmse, train_r2, test_mse, test_mae, test_rmse, test_r2, best_para, fi)
                elif i == "Random Search":
                    start = time.time()
                    model_name, model, parameters = op.regmodel(inputs,mod_i)
                    opt_method,train_mse, train_mae, train_rmse,  train_r2, test_mse, test_mae, test_rmse,  test_r2, best_para, tuning_time,test_time, fi = op.random(model_name,inputs,model,parameters,x_train,y_train,x_test,y_test)
                    end = time.time()
                    time = end-start
                    files.updatefilereg(inputs,result_file,model_name, opt_method,time, tuning_time,test_time, train_mse, train_mae, train_rmse,train_r2, test_mse, test_mae, test_rmse,  test_r2, best_para, fi)
                elif i == "Bayes Opt":
                    start = time.time()
                    opt_method,model_name, train_mse, train_mae, train_rmse, train_r2, test_mse, test_mae, test_rmse,  test_r2, best_para, tuning_time,test_time, fi = op.bayesreg(inputs,x_train,y_train,x_test,y_test)
                    end = time.time()
                    time = end-start
                    files.updatefilereg(inputs,result_file,model_name, opt_method,time, tuning_time,test_time, train_mse, train_mae, train_rmse, train_r2, test_mse, test_mae, test_rmse,  test_r2, best_para, fi)
            elif i == "Customized":
                start = time.time()
                parameters, model_name, train_mse, train_mae, train_rmse, train_r2, test_mse, test_mae, test_rmse,  test_r2, test_time, fi  = reg.gradientBoosting(inputs, x_train, y_train, x_test, y_test) 
                end = time.time()
                time = end-start
                files.updatefilereg(inputs,result_file,model_name,"None", time, "None", test_time, train_mse, train_mae, train_rmse,  train_r2, test_mse, test_mae, test_rmse,  test_r2, parameters, fi)
      
        elif mod_i == "AdaBoost":
            i = inputs.at[0,'Hyperparameters']
            if i == "DefaultValues":
                start = time.time()
                parameters, model_name, train_mse, train_mae, train_rmse, train_r2, test_mse, test_mae, test_rmse, test_r2, test_time, fi = reg.adaboost(inputs, x_train, y_train, x_test, y_test) 
                end = time.time()
                time = end-start
                files.updatefilereg(inputs,result_file,model_name,"None", time,"None", test_time, train_mse, train_mae, train_rmse, train_r2, test_mse, test_mae, test_rmse,  test_r2, parameters, fi)
            elif i == "Tune":
                i = inputs.at[0,'Tune']
                if i == "Grid Search":
                    start = time.time()
                    model_name, model, parameters = op.regmodel(inputs,mod_i)
                    opt_method, train_mse, train_mae, train_rmse, train_r2, test_mse, test_mae, test_rmse, test_r2, best_para, tuning_time, test_time, fi = op.grid(model_name, inputs,model,parameters,x_train,y_train,x_test,y_test)
                    end = time.time()
                    time = end-start
                    files.updatefilereg(inputs,result_file,model_name, opt_method,time, tuning_time, test_time, train_mse, train_mae, train_rmse, train_r2, test_mse, test_mae, test_rmse, test_r2, best_para, fi)
                elif i == "Random Search":
                    start = time.time()
                    model_name, model, parameters = op.regmodel(inputs,mod_i)
                    opt_method,train_mse, train_mae, train_rmse,  train_r2, test_mse, test_mae, test_rmse,  test_r2, best_para, tuning_time, test_time, fi = op.random(model_name,inputs,model,parameters,x_train,y_train,x_test,y_test)
                    end = time.time()
                    time = end-start
                    files.updatefilereg(inputs,result_file,model_name, opt_method,time, tuning_time, test_time, train_mse, train_mae, train_rmse,train_r2, test_mse, test_mae, test_rmse,  test_r2, best_para, fi)
                elif i == "Bayes Opt":
                    start = time.time()
                    opt_method,model_name, train_mse, train_mae, train_rmse, train_r2, test_mse, test_mae, test_rmse,  test_r2, best_para, tuning_time,test_time, fi = op.bayesreg(inputs,x_train,y_train,x_test,y_test)
                    end = time.time()
                    time = end-start
                    files.updatefilereg(inputs,result_file,model_name, opt_method,time, tuning_time,test_time, train_mse, train_mae, train_rmse, train_r2, test_mse, test_mae, test_rmse,  test_r2, best_para, fi)
            elif i == "Customized":
                start = time.time()
                parameters, model_name, train_mse, train_mae, train_rmse, train_r2, test_mse, test_mae, test_rmse,  test_r2, test_time, fi  = reg.adaboost(inputs, x_train, y_train, x_test, y_test) 
                end = time.time()
                time = end-start
                files.updatefilereg(inputs,result_file,model_name,"None", time, "None", test_time, train_mse, train_mae, train_rmse,  train_r2, test_mse, test_mae, test_rmse,  test_r2, parameters, fi)
        
        elif mod_i == "XgBoost":
            i = inputs.at[0,'Hyperparameters']
            if i == "DefaultValues":
                start = time.time()
                parameters, model_name, train_mse, train_mae, train_rmse, train_r2, test_mse, test_mae, test_rmse, test_r2, test_time, fi = reg.xgboost(inputs, x_train, y_train, x_test, y_test) 
                end = time.time()
                time = end-start
                files.updatefilereg(inputs,result_file,model_name,"None", time,"None", test_time, train_mse, train_mae, train_rmse, train_r2, test_mse, test_mae, test_rmse,  test_r2, parameters, fi)
            elif i == "Tune":
                i = inputs.at[0,'Tune']
                if i == "Grid Search":
                    start = time.time()
                    model_name, model, parameters = op.regmodel(inputs,mod_i)
                    opt_method, train_mse, train_mae, train_rmse, train_r2, test_mse, test_mae, test_rmse, test_r2, best_para, tuning_time, test_time, fi = op.grid(model_name, inputs,model,parameters,x_train,y_train,x_test,y_test)
                    end = time.time()
                    time = end-start
                    files.updatefilereg(inputs,result_file,model_name, opt_method,time, tuning_time, test_time, train_mse, train_mae, train_rmse, train_r2, test_mse, test_mae, test_rmse, test_r2, best_para, fi)
                elif i == "Random Search":
                    start = time.time()
                    model_name, model, parameters = op.regmodel(inputs,mod_i)
                    opt_method,train_mse, train_mae, train_rmse,  train_r2, test_mse, test_mae, test_rmse,  test_r2, best_para, tuning_time, test_time, fi = op.random(model_name,inputs,model,parameters,x_train,y_train,x_test,y_test)
                    end = time.time()
                    time = end-start
                    files.updatefilereg(inputs,result_file,model_name, opt_method,time, tuning_time, test_time, train_mse, train_mae, train_rmse,train_r2, test_mse, test_mae, test_rmse,  test_r2, best_para, fi)
                elif i == "Bayes Opt":
                    start = time.time()
                    opt_method,model_name, train_mse, train_mae, train_rmse, train_r2, test_mse, test_mae, test_rmse,  test_r2, best_para, tuning_time,test_time, fi = op.bayesreg(inputs,x_train,y_train,x_test,y_test)
                    end = time.time()
                    time = end-start
                    files.updatefilereg(inputs,result_file,model_name, opt_method,time, tuning_time,test_time, train_mse, train_mae, train_rmse, train_r2, test_mse, test_mae, test_rmse,  test_r2, best_para, fi)
            elif i == "Customized":
                start = time.time()
                parameters, model_name, train_mse, train_mae, train_rmse, train_r2, test_mse, test_mae, test_rmse,  test_r2,test_time,fi  = reg.xgboost(inputs, x_train, y_train, x_test, y_test) 
                end = time.time()
                time = end-start
                files.updatefilereg(inputs,result_file,model_name,"None", time, "None", test_time, train_mse, train_mae, train_rmse,  train_r2, test_mse, test_mae, test_rmse,  test_r2, parameters,fi)

        elif mod_i == "RandomForest":
            i = inputs.at[0,'Hyperparameters']
            if i == "DefaultValues":
                start = time.time()
                parameters, model_name, train_mse, train_mae, train_rmse, train_r2, test_mse, test_mae, test_rmse, test_r2, test_time, fi = reg.rf(inputs, x_train, y_train, x_test, y_test) 
                end = time.time()
                time = end-start
                files.updatefilereg(inputs,result_file,model_name,"None", time,"None", test_time, train_mse, train_mae, train_rmse, train_r2, test_mse, test_mae, test_rmse,  test_r2, parameters, fi)
            elif i == "Tune":
                i = inputs.at[0,'Tune']
                if i == "Grid Search":
                    start = time.time()
                    model_name, model, parameters = op.regmodel(inputs,mod_i)
                    opt_method, train_mse, train_mae, train_rmse, train_r2, test_mse, test_mae, test_rmse, test_r2, best_para, tuning_time, test_time, fi = op.grid(model_name, inputs,model,parameters,x_train,y_train,x_test,y_test)
                    end = time.time()
                    time = end-start
                    files.updatefilereg(inputs,result_file,model_name, opt_method,time, tuning_time, test_time, train_mse, train_mae, train_rmse, train_r2, test_mse, test_mae, test_rmse, test_r2, best_para, fi)
                elif i == "Random Search":
                    start = time.time()
                    model_name, model, parameters = op.regmodel(inputs,mod_i)
                    opt_method,train_mse, train_mae, train_rmse,  train_r2, test_mse, test_mae, test_rmse,  test_r2, best_para, tuning_time, test_time, fi = op.random(model_name,inputs,model,parameters,x_train,y_train,x_test,y_test)
                    end = time.time()
                    time = end-start
                    files.updatefilereg(inputs,result_file,model_name, opt_method,time, tuning_time,test_time, train_mse, train_mae, train_rmse,train_r2, test_mse, test_mae, test_rmse,  test_r2, best_para, fi)
                elif i == "Bayes Opt":
                    start = time.time()
                    opt_method,model_name, train_mse, train_mae, train_rmse, train_r2, test_mse, test_mae, test_rmse,  test_r2, best_para, tuning_time,test_time, fi = op.bayesreg(inputs,x_train,y_train,x_test,y_test)
                    end = time.time()
                    time = end-start
                    files.updatefilereg(inputs,result_file,model_name, opt_method,time, tuning_time,test_time, train_mse, train_mae, train_rmse, train_r2, test_mse, test_mae, test_rmse,  test_r2, best_para, fi)
            elif i == "Customized":
                start = time.time()
                parameters, model_name, train_mse, train_mae, train_rmse, train_r2, test_mse, test_mae, test_rmse,  test_r2, test_time, fi  = reg.rf(inputs, x_train, y_train, x_test, y_test) 
                end = time.time()
                time = end-start
                files.updatefilereg(inputs,result_file,model_name,"None", time, "None", test_time, train_mse, train_mae, train_rmse,  train_r2, test_mse, test_mae, test_rmse,  test_r2, parameters, fi)
     
        elif mod_i == "LinearRegression":
            i = inputs.at[0,'Hyperparameters']
            if i == "DefaultValues":
                start = time.time()
                parameters, model_name, train_mse, train_mae, train_rmse, train_r2, test_mse, test_mae, test_rmse, test_r2, test_time, fi = reg.lr(inputs, x_train, y_train, x_test, y_test) 
                end = time.time()
                time = end-start
                files.updatefilereg(inputs,result_file,model_name,"None", time,"None", test_time, train_mse, train_mae, train_rmse, train_r2, test_mse, test_mae, test_rmse,  test_r2, parameters, fi)
            elif i == "Tune":
                print("Doesn't exist")
            elif i == "Customized":
                print("Doesn't exist")
        else:
            print("Unknown ML category, please recheck the input form.") 
    
            

        
       