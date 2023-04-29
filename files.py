#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 12:33:21 2020

@author: HarounJr
"""

def openfile():
    import csv
    out_file1 = 'ResultSummary.csv'
    of_connection1 = open(out_file1, 'w')
    writer = csv.writer(of_connection1)
    writer.writerow(["Setting","File Name","Input Variables","Target Variable","MissingValuesStategy",'Test Size','Random State','Stratified Sampling','ML Method','Optimization Method','CV Folds','Iterations','Time taken','Tuning time','test time','Training Error (MSE)','Training Error (MAE)','Training Error (RMSE)','Training Error (R2)','Testing Error (MSE)','Testing Error (MAE)','Testing Error (RMSE)','Testing Error (R2)', 'Train Accuracy','Train recall', 'Train precision','Train F1','Test Accuracy','Test Recall','Test Precision','Test F1','Optimal Parameters', 'FI'])
    of_connection1.close()
    return out_file1

def updatefilereg(inputs, out_file1, model, opt_model, time, tuning_time, test_time, train_mse, train_mae,train_rmse,  train_r2, test_mse,test_mae,test_rmse,  test_r2, parameters, fi):
    import csv
    strategy = inputs['Strategy'].values[0]
    testsize = inputs['TestSize'].values[0]
    randomstate = int(inputs['RandomState'].values[0])
    ssampling = inputs['StratifiedSampling'].values[0]
    setting = int(inputs['Setting'].values[0])
    xs = inputs['Xs'].values[0]
    ys = inputs['Y'].values[0]
    datafile = inputs['Data File'].values[0]
    fold = int(inputs['CVFolds'].values[0])
    if inputs.at[0,'Tune'] == "Grid Search":
        iterations = "Full Grid"
    elif inputs.at[0,'Hyperparameters'] == "DefaultValues":
        iterations = "-"
    else:
        iterations = int(inputs['Iterations'].values[0])
        
    if tuning_time == "None":
        tuning_time  = 0
    of_connection1 = open(out_file1, 'a')
    writer = csv.writer(of_connection1)
    writer.writerow([setting,datafile,xs,ys,strategy,testsize,randomstate,ssampling,model,opt_model,fold, iterations,round(float(time), 4),round(tuning_time, 4), round(test_time, 4), round(float(train_mse),4),round(float(train_mae),4),round(float(train_rmse),4),round(float(train_r2),4),round(float(test_mse),4),round(float(test_mae),4),round(float(test_rmse),4),round(float(test_r2),4),"N/A","N/A","N/A","N/A","N/A","N/A","N/A","N/A", parameters, fi])
    of_connection1.close()
    return

def updatefilecl(inputs, out_file1, model, opt_model,time, tuning_time, test_time, train_accuracy, train_recall, train_precision, train_f1, test_accuracy,test_recall, test_precision, test_f1, parameters, fi):
    import csv
    strategy = inputs['Strategy'].values[0]
    testsize = inputs['TestSize'].values[0]
    randomstate = int(inputs['RandomState'].values[0])
    ssampling = inputs['StratifiedSampling'].values[0]
    setting = int(inputs['Setting'].values[0])
    xs = inputs['Xs'].values[0]
    ys = inputs['Y'].values[0]
    datafile = inputs['Data File'].values[0]

    fold = int(inputs['CVFolds'].values[0])
    if inputs.at[0,'Tune'] == "Grid Search":
        iterations = "Full Grid"
    elif inputs.at[0,'Hyperparameters'] == "DefaultValues" or inputs.at[0,'Hyperparameters'] == "Customized":
        iterations = "-"  
    else:
        iterations = int(inputs['Iterations'].values[0])
    if tuning_time == "None":
        tuning_time  = 0
    of_connection1 = open(out_file1, 'a')
    writer = csv.writer(of_connection1)
    writer.writerow([setting,datafile,xs,ys,strategy,testsize,randomstate,ssampling,model,opt_model,fold, iterations,round(time, 4),round(tuning_time, 4),round(test_time, 4), "N/A","N/A","N/A","N/A","N/A","N/A","N/A","N/A",round(train_accuracy,4),round(train_recall,4),round(train_precision,4),round(train_f1,4),round(test_accuracy,4),round(test_recall,4),round(test_precision,4),round(test_f1,4),parameters, fi])
    of_connection1.close()
    return