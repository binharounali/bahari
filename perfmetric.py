#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 31 19:00:13 2020

@author: HarounJr
"""

def perfreg(inputs, model, train_mse,train_mae, train_rmse, train_r2,  x_train, y_train, x_test, y_test):
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import mean_absolute_error
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import r2_score
    import math
    import time
    
    total_mse =0
    total_mae =0
    total_rmse =0
    total_r2 = 0

    loops = 1
    for i in range(loops):
       start = time.time()
       y_pred = model.predict(x_test)
       end = time.time()
       test_time = end -start
       mse =mean_squared_error(y_test, y_pred)
       mae =mean_absolute_error(y_test, y_pred)
       rmse = math.sqrt(mse)
       r2 =r2_score(y_test, y_pred)
       
       total_mse = total_mse + mse
       total_mae = total_mae + mae
       total_rmse = total_rmse + rmse
       total_r2 = total_r2 + r2
       model.fit(x_train, y_train)
       
    test_mse = total_mse/loops
    test_mae = total_mae/loops
    test_rmse = total_rmse/loops
    test_r2 = total_r2/loops
    
    print("               ***              ")
    print("Training error (MSE):", train_mse)
    print("Testing error (MSE):", test_mse)
    print("               ***              ")
    print("Training error (MAE):", train_mae)
    print("Testing error (MAE):", test_mae)
    print("               ***              ")
    print("Training error (RMSE):", train_rmse)
    print("Testing error (RMSE):", test_rmse)
    print("               ***              ")
    print("Training error (R2):", train_r2)
    print("Testing error (R2):", test_r2)
    print("               ***              ")

    return train_mse, train_mae, train_rmse, train_r2, test_mse, test_mae, test_rmse, test_r2, test_time

def perfregnew(inputs, model, train_mse,train_mae, train_rmse, train_r2):
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import mean_absolute_error
    from sklearn.metrics import r2_score
    import pandas as pd
    import preprocess as pp
    import math
    import time

    
    file = inputs.at[0,'NewDataFile']
    df = pd.read_excel (file)
    
    xs = inputs['Xs'].values[0].split(",")
    ys = inputs.at[0,'Y']
    x = pd.DataFrame(df, columns= xs)
    y = pd.DataFrame(df, columns= [ys])

    #Data Imputation
    x,y = pp.impute(inputs,x,y)
 
    #Data Splitting
    x_train, x_test, y_train, y_test = pp.datasplit(inputs,x,y)
    
    total_mse =0
    total_mae =0
    total_rmse =0
    total_r2 = 0

    loops = 1
    for i in range(loops):
       start = time.time()
       y_pred = model.predict(x_test)
       end = time.time()
       test_time = end -start 
        
       mse =mean_squared_error(y_test, y_pred)
       mae =mean_absolute_error(y_test, y_pred)
       rmse =  math.sqrt(mse)
       r2 =r2_score(y_test, y_pred)
       total_mse = total_mse + mse
       total_mae = total_mae + mae
       total_rmse = total_rmse + rmse
       total_r2 = total_r2 + r2
       model.fit(x_train, y_train)
       
    test_mse = total_mse/loops
    test_mae = total_mae/loops
    test_rmse = total_rmse/loops
    test_r2 = total_r2/loops

    print("               ***              ")
    print("Training error (MSE):", train_mse)
    print("Testing error (MSE):", test_mse)
    print("               ***              ")
    print("Training error (MAE):", train_mae)
    print("Testing error (MAE):", test_mae)
    print("               ***              ")
    print("Training error (RMSE):", train_rmse)
    print("Testing error (RMSE):", test_rmse)
    print("               ***              ")
    print("Training error (R2):", train_r2)
    print("Testing error (R2):", test_r2)
    
    return train_mse, train_mae, train_rmse, train_r2, test_mse, test_mae, test_rmse,  test_r2, test_time

def perfcl(inputs,model,train_accuracy,train_recall,train_precision,train_f1,x_train, y_train, x_test, y_test):
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import f1_score
    from sklearn.metrics import confusion_matrix
    import seaborn as sn
    import matplotlib.pyplot as plt
    import time
    
    total_accuracy = 0
    total_recall = 0
    total_precision = 0
    total_f1 = 0
    loops = 1
    for i in range(loops):
        start = time.time()
        y_pred = model.predict(x_test)
        end = time.time()
        test_time = end -start
        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred, average = 'weighted')
        precision = precision_score(y_test, y_pred, average = 'weighted')
        f1 = f1_score(y_test, y_pred, average = 'weighted')
        total_accuracy = total_accuracy + accuracy
        total_recall = total_recall + recall
        total_precision = total_precision + precision
        total_f1 = total_f1 + f1
        
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        sn.set(font_scale=1) # for label size
        target = str(inputs.at[0,'Y'])
        method_name = str(inputs.at[0,'Methods'])
        ax= plt.subplot()
        sn.heatmap(cm, annot=True, cbar = False, cmap="YlGnBu", square = True, linewidths = 0.5) # font size
        plt.title(target + '_' + method_name +'.')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(target + '_' + method_name +' .png', dpi = 900)
        plt.show()
        
        model.fit(x_train, y_train)
    
    test_accuracy = total_accuracy/loops
    test_recall = total_recall/loops
    test_precision = total_precision/loops
    test_f1 = total_f1/loops
    
    print("               ***              ")
    print("Training accuracy:", round(train_accuracy,4))
    print("Testing accuracy:", round(test_accuracy,4))
    print("               ***              ")
    print("Training recall:", round(train_recall,4))
    print("Testing recall:", round(test_recall,4))
    print("               ***              ")
    print("Training precision:", round(train_precision,4))
    print("Testing precision:", round(test_precision,4))
    print("               ***              ")
    print("Training f1:", round(train_f1,4))
    print("Testing f1:", round(test_f1,4))
    print("                            ")

    return train_accuracy, train_recall,train_precision,train_f1, test_accuracy, test_recall, test_precision, test_f1, test_time

def perfclnew(inputs,model,train_accuracy, train_recall,train_precision,train_f1):
    import pandas as pd
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import f1_score
    from sklearn.metrics import confusion_matrix
    import preprocess as pp
    import seaborn as sn
    import matplotlib.pyplot as plt
    import time

    file = inputs.at[0,'NewDataFile']
    df = pd.read_excel (file)
    
    xs = inputs['Xs'].values[0].split(",")
    ys = inputs.at[0,'Y']
    x = pd.DataFrame(df, columns= xs)
    y = pd.DataFrame(df, columns= [ys])

    #Data Imputation
    x,y = pp.impute(inputs,x,y)
 
    #Data Splitting
    x_train, x_test, y_train, y_test = pp.datasplit(inputs,x,y)
  
    total_accuracy = 0
    total_recall = 0
    total_precision = 0
    total_f1 = 0
    loops = 1
    for i in range(loops):
        start = time.time()
        y_pred = model.predict(x_test)
        end = time.time()
        test_time = end -start
        
        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred, average = 'weighted')
        precision = precision_score(y_test, y_pred, average = 'weighted')
        f1 = f1_score(y_test, y_pred, average = 'weighted')
        
        total_accuracy = total_accuracy + accuracy
        total_recall = total_recall + recall
        total_precision = total_precision + precision
        total_f1 = total_f1 + f1
        
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        sn.set(font_scale=1) # for label size
        target = inputs.at[0,'Target']
        method_name = inputs.at[0,'Methods']
        case = str(inputs.at[0,'Cases'])
        ax= plt.subplot()
        sn.heatmap(cm, annot=True, cbar = False, cmap="YlGnBu", square = True, linewidths = 0.5) # font size
        plt.title(target + '_' + method_name +'.')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(target + '_' + method_name +' .png', dpi = 900)
        plt.show()
    
    test_accuracy = total_accuracy/loops
    test_recall = total_recall/loops
    test_precision = total_precision/loops
    test_f1 = total_f1/loops
    
    print("               ***              ")
    print("Training accuracy:", round(train_accuracy,4))
    print("Testing accuracy:", round(test_accuracy,4))
    print("               ***              ")
    print("Training recall:", round(train_recall,4))
    print("Testing recall:", round(test_recall,4))
    print("               ***              ")
    print("Training precision:", round(train_precision,4))
    print("Testing precision:", round(test_precision,4))
    print("               ***              ")
    print("Training f1:", round(train_f1,4))
    print("Testing f1:", round(test_f1,4))
    print("                            ")

    return train_accuracy, train_recall,train_precision,train_f1, test_accuracy, test_recall, test_precision, test_f1, test_time

