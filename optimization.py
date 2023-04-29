#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 22:46:45 2020

@author: HarounJr
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 15:24:01 2020

@author: ea.lab
"""

##!/usr/bin/env python3
## -*- coding: utf-8 -*-
#"""
#Created on Thu Feb  6 11:14:31 2020
#
#@author: HarounJr

def regmodel(inputs,i):
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.ensemble import AdaBoostRegressor
    import xgboost as xgb
    from sklearn.ensemble import RandomForestRegressor
    
    if i == "GradientBoosting":
        seed = int(inputs.at[0,'RandomState'])
        if seed != 0:
            r_state = seed
        else:
            r_state = None
        model = GradientBoostingRegressor(random_state=r_state)
        max_depth = inputs['gb_md'].values[0].split(";")
        max_depth = list(map(int, max_depth)) 
        learning_rate = inputs['gb_lr'].values[0].split(";")
        learning_rate = list(map(float, learning_rate))
        min_samples_split = inputs['gb_mss'].values[0].split(";")
        min_samples_split = list(map(int, min_samples_split))
        n_estimators = inputs['gb_ne'].values[0].split(";")
        n_estimators = list(map(int, n_estimators)) 
        parameters ={'n_estimators':n_estimators,
                     'max_depth':max_depth,
                     'learning_rate' : learning_rate,
                     'min_samples_split' : min_samples_split
                     }
        
    if i == "AdaBoost":
        seed = int(inputs.at[0,'RandomState'])
        if seed != 0:
            r_state = seed
        else:
            r_state = None
        model = AdaBoostRegressor(random_state=r_state)
        learning_rate = inputs['ada_lr'].values[0].split(";")
        learning_rate = list(map(float, learning_rate))
        loss = inputs['ada_loss'].values[0].split(";")
        loss = list(map(str, loss))
        n_estimators = inputs['ada_ne'].values[0].split(";")
        n_estimators = list(map(int, n_estimators)) 
        parameters ={'n_estimators':n_estimators,
                     'loss':loss,
                     'learning_rate' : learning_rate,
                     }

    if i == "RandomForest":
        seed = int(inputs.at[0,'RandomState'])
        if seed != 0:
            r_state = seed
        else:
            r_state = None
        model = RandomForestRegressor(random_state=r_state)
        max_features = str(inputs['rf_mf'].values[0]).split(";")
        max_features = list(map(str, max_features)) 
        min_samples_leaf = str(inputs['rf_msl'].values[0]).split(";")
        min_samples_leaf = list(map(int, min_samples_leaf)) 
        n_estimators = str(inputs['rf_ne'].values[0]).split(";")
        n_estimators = list(map(int, n_estimators)) 
        parameters ={'n_estimators':n_estimators,
                     'max_features':max_features,
                     'min_samples_leaf' : min_samples_leaf
                     }
        
    if i == "XgBoost":
        seed = int(inputs.at[0,'RandomState'])
        if seed != 0:
            r_state = seed
        else:
            r_state = None
        model = xgb.XGBRegressor(random_state=r_state)
        max_depth = str(inputs['xgb_md'].values[0]).split(";")
        max_depth = list(map(int, max_depth)) 
        learning_rate = str(inputs['xgb_lr'].values[0]).split(";")
        learning_rate = list(map(float, learning_rate)) 
        n_estimators = str(inputs['xgb_ne'].values[0]).split(";")
        n_estimators = list(map(int, n_estimators)) 
        parameters ={'n_estimators':n_estimators,
                     'max_depth':max_depth,
                     'learning_rate' : learning_rate
                     }
    return i, model, parameters

def grid(model_name,inputs,model,parameters,x_train,y_train,x_test,y_test):
    from sklearn.model_selection import GridSearchCV
    import perfmetric as pm
    import time
    
    folds = int(inputs.at[0,'CVFolds'])
    metric= inputs.at[0,'Metric']
    start = time.time()
    grid_search = GridSearchCV(model, param_grid =parameters, cv=folds, scoring =['neg_mean_squared_error','neg_mean_absolute_error','neg_root_mean_squared_error','r2'],refit = metric,verbose = 10)
    grid_search.fit(x_train, y_train)
    end = time.time()       
    time_taken = end - start
   
    best_para=  grid_search.best_params_
    best_model = grid_search.best_estimator_
    train_mse = ((grid_search.cv_results_['mean_test_neg_mean_squared_error'][grid_search.best_index_]) * -1)
    train_mae = ((grid_search.cv_results_['mean_test_neg_mean_absolute_error'][grid_search.best_index_]) * -1)
    train_rmse = ((grid_search.cv_results_['mean_test_neg_root_mean_squared_error'][grid_search.best_index_]) * -1)
    train_r2 = (grid_search.cv_results_['mean_test_r2'][grid_search.best_index_]) 
    fi =best_model.feature_importances_

    j = inputs.at[0,'ModelOptions']
    if j == "Test with New Data":
        train_mse, train_mae, train_rmse, train_r2, test_mse, test_mae, test_rmse, test_r2  = pm.perfregnew(inputs,best_model,train_mse, train_mae, train_rmse, train_r2)
    else:
        train_mse, train_mae, train_rmse, train_r2, test_mse, test_mae, test_rmse, test_r2, test_time = pm.perfreg(inputs,best_model, train_mse, train_mae, train_rmse, train_r2, x_train, y_train, x_test, y_test)
    return "Grid Search", train_mse, train_mae, train_rmse,  train_r2, test_mse, test_mae, test_rmse, test_r2, best_para,time_taken, test_time, fi


def random(model_name,inputs,model,parameters,x_train,y_train,x_test,y_test):
    
    from sklearn.model_selection import RandomizedSearchCV
    import perfmetric as pm
    import time
    import pandas as pd
   
    seed = int(inputs.at[0,'RandomState'])
    if seed != 0:
        r_state = seed
    else:
        r_state = None
        
    folds = int(inputs.at[0,'CVFolds'])
    iterations = int(inputs.at[0,'Iterations'])
    metric= inputs.at[0,'Metric']
    start = time.time()
    random_search = RandomizedSearchCV(model, param_distributions =parameters, cv=folds, scoring =['neg_mean_squared_error','neg_mean_absolute_error','neg_root_mean_squared_error','r2'],refit = metric, n_iter=iterations, verbose = 10, random_state = r_state)
    random_search.fit(x_train, y_train)
    end = time.time()       
    time_taken = end - start
   
    best_para= random_search.best_params_
    best_model = random_search.best_estimator_
    train_mse = ((random_search.cv_results_['mean_test_neg_mean_squared_error'][random_search.best_index_]) * -1)
    train_mae = ((random_search.cv_results_['mean_test_neg_mean_absolute_error'][random_search.best_index_]) * -1)
    train_rmse = ((random_search.cv_results_['mean_test_neg_root_mean_squared_error'][random_search.best_index_]) * -1)
    train_r2 = (random_search.cv_results_['mean_test_r2'][random_search.best_index_]) 
    fi =best_model.feature_importances_
    
    j = inputs.at[0,'ModelOptions']
    if j == "Test with New Data":
        train_mse, train_mae, train_rmse,  train_r2, test_mse, test_mae, test_rmse,  test_r2  = pm.perfregnew(inputs,best_model,train_mse, train_mae, train_rmse,  train_r2)
    else:
        train_mse, train_mae, train_rmse,  train_r2, test_mse, test_mae, test_rmse,  test_r2, test_time = pm.perfreg(inputs,best_model, train_mse, train_mae, train_rmse,  train_r2, x_train, y_train, x_test, y_test)
    return "Random Search", train_mse, train_mae, train_rmse,  train_r2, test_mse, test_mae, test_rmse, test_r2, best_para,time_taken, test_time,fi
 
def bayesreg(inputs,x_train,y_train,x_test,y_test):
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.ensemble import AdaBoostRegressor
    import xgboost as xgb
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import cross_val_score
    from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
    import perfmetric as pm
    import numpy as np
    import time
    
    j = int(inputs.at[0,'CVFolds'])
    mod = (inputs.at[0,'Methods'])
    def objective(params):
        if mod == "GradientBoosting":
            params['n_estimators'] = int(params['n_estimators'])
            params['min_samples_split'] = int(params['min_samples_split'])
            params['max_depth'] = int(params['max_depth'])
            model = GradientBoostingRegressor(**params)
        if mod == "RandomForest":
            params['n_estimators'] = int(params['n_estimators'])
            params['min_samples_leaf'] = int(params['min_samples_leaf'])
            model = RandomForestRegressor(**params)
        if mod == "AdaBoost":
            params['n_estimators'] = int(params['n_estimators'])
            model = AdaBoostRegressor(**params)
        if mod == "XgBoost":
            params['n_estimators'] = int(params['n_estimators'])
            params['max_depth'] = int(params['max_depth'])
            model = xgb.XGBRegressor(**params)
        mse_neg = cross_val_score(model, x_train, y_train, scoring = 'neg_mean_absolute_error', cv=j).mean()
        mse_train_BO = mse_neg*-1            
        return {'loss': mse_train_BO, 'params': params, 'status': STATUS_OK}
    if mod == "GradientBoosting":
        blr = inputs['gb_b_lr'].values[0].split(",")
        blr = list(map(float, blr))
        bmd = inputs['gb_b_md'].values[0].split(",")
        bmd = list(map(int, bmd)) 
        bne = inputs['gb_b_ne'].values[0].split(",")
        bne = list(map(int, bne)) 
        bmss = inputs['gb_b_mss'].values[0].split(",")
        bmss = list(map(int, bmss)) 
        space = {'learning_rate': hp.loguniform('learning_rate', np.log(blr[0]),np.log(blr[1])),
                 'max_depth': hp.quniform('max_depth', bmd[0],bmd[1],bmd[2]),
                 'n_estimators': hp.quniform('n_estimators',bne[0],bne[1],bne[2] ),
                 'min_samples_split': hp.quniform('min_samples_split', bmss[0],bmss[1],bmss[2])
                } 
    if mod == "RandomForest":
        bmf = inputs['rf_b_mf'].values[0].split(",")
        bmf = list(map(str, bmf)) 
        bne = inputs['rf_b_ne'].values[0].split(",")
        bne = list(map(int, bne)) 
        bmsl = inputs['rf_b_msl'].values[0].split(",")
        bmsl = list(map(int, bmsl)) 
        space = {'max_features': hp.choice('max_features', bmf),
                 'n_estimators': hp.quniform('n_estimators',bne[0],bne[1],bne[2] ),
                 'min_samples_leaf': hp.quniform('min_samples_leaf', bmsl[0],bmsl[1],bmsl[2])
                } 
    if mod == "AdaBoost":
        blr = inputs['ada_b_lr'].values[0].split(",")
        blr = list(map(float, blr))
        bne = inputs['ada_b_ne'].values[0].split(",")
        bne = list(map(int, bne)) 
        bloss = inputs['ada_b_loss'].values[0].split(",")
        bloss = list(map(str, bloss)) 
        space = {'learning_rate': hp.loguniform('learning_rate', np.log(blr[0]),np.log(blr[1])),
                 'n_estimators': hp.quniform('n_estimators',bne[0],bne[1],bne[2] ),
                 'loss': hp.choice('loss', bloss)
                } 
    if mod == "XgBoost":
        blr = inputs['xgb_b_lr'].values[0].split(",")
        blr = list(map(float, blr))
        bne = inputs['xgb_b_ne'].values[0].split(",")
        bne = list(map(int, bne)) 
        bmd = inputs['xgb_b_md'].values[0].split(",")
        bmd = list(map(int, bmd)) 
        space = {'learning_rate': hp.loguniform('learning_rate', np.log(blr[0]),np.log(blr[1])),
                 'n_estimators': hp.quniform('n_estimators',bne[0],bne[1],bne[2] ),
                 'max_depth': hp.quniform('max_depth', bmd[0],bmd[1],bmd[2]),
                } 
            
    iteration = int(inputs.at[0,'Iterations'])
    i = inputs.at[0,'TPE']
    if i == "Suggest":
        tpe_algorithm = tpe.suggest
        trials = Trials()
    if i == "Random":
        tpe_algorithm = tpe.rand.suggest
        trials = Trials()
        
    start = time.time()
    best = fmin(fn = objective, space = space, algo = tpe_algorithm, max_evals = iteration, trials=trials)
    end = time.time()
    tuning_time = end - start
    
    mse_list= trials.losses()
    train_mse = min(mse_list)
    
    print(space_eval(space, best))
    best_para_BO = space_eval(space, best)
    
    if mod == "GradientBoosting":
        model = GradientBoostingRegressor(
                            max_depth=int(best_para_BO['max_depth']),
                            n_estimators=int(best_para_BO['n_estimators']),
                            learning_rate=best_para_BO['learning_rate'],
                            min_samples_split =int(best_para_BO['min_samples_split'])
                            ) 
    if mod == "RandomForest":
        model = RandomForestRegressor(
                            max_features= best_para_BO['max_features'],
                            n_estimators=int(best_para_BO['n_estimators']),
                            min_samples_leaf =int(best_para_BO['min_samples_leaf'])
                            ) 
    if mod == "AdaBoost":
        model = AdaBoostRegressor(
                            n_estimators=int(best_para_BO['n_estimators']),
                            learning_rate=best_para_BO['learning_rate'],
                            loss =best_para_BO['loss']
                            ) 
    if mod == "XgBoost":
        model =model = xgb.XGBRegressor(
                            n_estimators=int(best_para_BO['n_estimators']),
                            learning_rate=best_para_BO['learning_rate'],
                            max_depth=int(best_para_BO['max_depth']),
                            )  
    model.fit(x_train, y_train)
    train_mse, train_mae, train_rmse, train_r2, test_mse,test_mae, test_rmse, test_r2, test_time = pm.perfreg(inputs,model, train_mse, 0, 0, 0, x_train, y_train, x_test, y_test)
    
        
    return "Bayesian Search", mod, train_mse, train_mae, train_rmse,  train_r2, test_mse, test_mae, test_rmse, test_r2, best_para_BO, tuning_time, test_time, "None"
    
def clmodel(inputs,i):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn import svm
    import mord as m
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.ensemble import AdaBoostClassifier
    import xgboost as xgb
 
    if i == "RandomForest":
        rs = int(inputs.at[0,'RandomState'])
        if rs != 0:
            frs = rs
        else:
            frs = None
        model = RandomForestClassifier(random_state = frs)
        max_depth = inputs['rf_md_cl'].values[0].split(",")
        max_depth = list(map(int, max_depth)) 
        min_samples_leaf = inputs['rf_msl_cl'].values[0].split(",")
        min_samples_leaf = list(map(int, min_samples_leaf)) 
        n_estimators = inputs['rf_ne_cl'].values[0].split(",")
        n_estimators = list(map(int, n_estimators)) 
        parameters ={'n_estimators':n_estimators,
                     'max_depth':max_depth,
                     'min_samples_leaf': min_samples_leaf
                     }
        
    if i == "SVM":
        rs = int(inputs.at[0,'RandomState'])
        if rs != 0:
            frs = rs
        else:
            frs = None
        model = svm.SVC(random_state = frs)
        kernel = inputs['svm_kernel_cl'].values[0].split(",")
        kernel = list(map(str, kernel)) 
        c = inputs['svm_c_cl'].values[0].split(",")
        c = list(map(float, c)) 
        parameters ={'kernel':kernel,
                     'C':c,
                     }
        
    if i == "Default_LR":
        rs = int(inputs.at[0,'RandomState'])
        if rs != 0:
            frs = rs
        else:
            frs = None
        model = LogisticRegression(solver = 'liblinear', random_state = frs)
        penalty = inputs['lr_penalty_cl'].values[0].split(",")
        penalty = list(map(str, penalty)) 
        c = inputs['lr_c_cl'].values[0].split(",")
        c = list(map(float, c)) 
        parameters ={'penalty':penalty,
                     'C':c,
                     }
        
    if i == "AdaBoost":
        rs = int(inputs.at[0,'RandomState'])
        if rs != 0:
            frs = rs
        else:
            frs = None
        model = AdaBoostClassifier(random_state = frs)
        learning_rate = inputs['ada_lr_cl'].values[0].split(",")
        learning_rate = list(map(float, learning_rate))
        n_estimators = inputs['ada_ne_cl'].values[0].split(",")
        n_estimators = list(map(int, n_estimators)) 
        algorithm = inputs['ada_algorithm_cl'].values[0].split(",")
        algorithm = list(map(str, algorithm)) 
        parameters ={'n_estimators':n_estimators,
                     'algorithm' : algorithm,
                     'learning_rate': learning_rate
                     }
        
    if i == "XgBoost":
        rs = int(inputs.at[0,'RandomState'])
        if rs != 0:
            frs = rs
        else:
            frs = None
        model = xgb.XGBClassifier()
        learning_rate = inputs['xgb_lr_cl'].values[0].split(",")
        learning_rate = list(map(float, learning_rate))
        n_estimators = inputs['xgb_ne_cl'].values[0].split(",")
        n_estimators = list(map(int, n_estimators)) 
        max_depth = inputs['xgb_md_cl'].values[0].split(",")
        max_depth = list(map(int, max_depth)) 
        parameters ={'n_estimators':n_estimators,
                     'max_depth' : max_depth,
                     'learning_rate': learning_rate
                     }
        
    if i == "GradientBoosting":
        rs = int(inputs.at[0,'RandomState'])
        if rs != 0:
            frs = rs
        else:
            frs = None
        model = GradientBoostingClassifier(random_state = frs)
        max_depth = inputs['gb_md_cl'].values[0].split(",")
        max_depth = list(map(int, max_depth)) 
        learning_rate = inputs['gb_lr_cl'].values[0].split(",")
        learning_rate = list(map(float, learning_rate))
        min_samples_split = inputs['gb_mss_cl'].values[0].split(",")
        min_samples_split = list(map(int, min_samples_split))
        n_estimators = inputs['gb_ne_cl'].values[0].split(",")
        n_estimators = list(map(int, n_estimators)) 
        parameters ={'n_estimators':n_estimators,
                     'max_depth':max_depth,
                     'learning_rate' : learning_rate,
                     'min_samples_split' : min_samples_split
                     }
    if i == "Ordinal_LR":
        model = m.LogisticIT()
        alpha = inputs['ord_alpha_cl'].values[0].split(",")
        alpha = list(map(float, alpha))
        
        parameters ={'alpha':alpha
                     }
        
    return i, model, parameters

def gridcl(model_name,inputs,model,parameters,x_train,y_train,x_test,y_test):
    from sklearn.model_selection import GridSearchCV
    import perfmetric as pm
    import numpy as np
    import time
    
    if inputs.at[0,'Methods'] == "Ordinal_LR":
        y_train = np.array(y_train, dtype=np.float64)
        y_train = y_train.astype(np.int)
        y_test = np.array(y_test, dtype=np.float64)
        y_test = y_test.astype(np.int) 
        
    folds = int(inputs.at[0,'CVFolds'])        
    metric = inputs.at[0,'Metric'] 
    
    start = time.time()
    grid_search = GridSearchCV(model, param_grid =parameters, cv=folds, scoring =['accuracy','recall_weighted','precision_weighted','f1_weighted'],refit = metric,verbose = 10)
    grid_search.fit(x_train, y_train) 
    end = time.time()       
    time_taken = end - start
    
    best_para= grid_search.best_params_
    best_model = grid_search.best_estimator_
        
    train_accuracy = (grid_search.cv_results_['mean_test_accuracy'][grid_search.best_index_]) 
    train_recall = (grid_search.cv_results_['mean_test_recall_weighted'][grid_search.best_index_]) 
    train_precision = (grid_search.cv_results_['mean_test_precision_weighted'][grid_search.best_index_]) 
    train_f1 = (grid_search.cv_results_['mean_test_f1_weighted'][grid_search.best_index_]) 
    print("Best parameters are:", best_para)

    if inputs.at[0,'Methods'] == "Ordinal_LR" or inputs.at[0,'Methods'] == "Default_LR":
        coef = grid_search.best_estimator_.coef_
        fi = coef
    elif inputs.at[0,'Methods'] == "SVM":
        fi="No FI for SVM in optimizing phase"
    else:
        fi =best_model.feature_importances_

    j = inputs.at[0,'ModelOptions']
    if j == "Test with New Data":
        train_accuracy, train_recall,train_precision,train_f1, test_accuracy, test_recall, test_precision, test_f1 = pm.perfclnew(inputs,best_model,train_accuracy)
    else:
        train_accuracy, train_recall,train_precision,train_f1, test_accuracy, test_recall, test_precision, test_f1, test_time= pm.perfcl(inputs,best_model, train_accuracy,train_recall,train_precision,train_f1, x_train, y_train, x_test, y_test)
    return "Grid Search", train_accuracy, train_recall,train_precision,train_f1, test_accuracy,test_recall, test_precision, test_f1, best_para,time_taken, test_time, fi
  
def randomcl(model_name,inputs,model,parameters,x_train,y_train,x_test,y_test):
    from sklearn.model_selection import RandomizedSearchCV
    import perfmetric as pm
    import numpy as np
    import time

    seed = int(inputs.at[0,'RandomState'])
    if seed != 0:            
      r_state = seed
    else:   
      r_state = None
      
    if inputs.at[0,'Methods'] == "Ordinal_LR":
        y_train = np.array(y_train, dtype=np.float64)
        y_train = y_train.astype(np.int)
        y_test = np.array(y_test, dtype=np.float64)
        y_test = y_test.astype(np.int) 
        
    folds = int(inputs.at[0,'CVFolds'])
    iterations = int(inputs.at[0,'Iterations'])
    metric = inputs.at[0,'Metric'] 
    
    start = time.time()
    random_search = RandomizedSearchCV(model, param_distributions =parameters, cv=folds, scoring =['accuracy','recall_weighted','precision_weighted','f1_weighted'], refit = metric, n_iter=iterations, verbose = 10, random_state = r_state)
    random_search.fit(x_train, y_train) 
    end = time.time()       
    time_taken = end - start
    
    best_para= random_search.best_params_
    best_model = random_search.best_estimator_
    train_accuracy = (random_search.cv_results_['mean_test_accuracy'][random_search.best_index_]) 
    train_recall = (random_search.cv_results_['mean_test_recall_weighted'][random_search.best_index_]) 
    train_precision = (random_search.cv_results_['mean_test_precision_weighted'][random_search.best_index_]) 
    train_f1 = (random_search.cv_results_['mean_test_f1_weighted'][random_search.best_index_]) 
    
    if inputs.at[0,'Methods'] == "Ordinal_LR" or inputs.at[0,'Methods'] == "Default_LR":
        coef = random_search.best_estimator_.coef_
        fi = coef
    elif inputs.at[0,'Methods'] == "SVM":
        fi="No FI for SVM in optimizing phase"
    else:
        fi =best_model.feature_importances_

    print("Best parameters are:", best_para)

    j = inputs.at[0,'ModelOptions']
    if j == "Test with New Data":
        train_accuracy, train_recall,train_precision,train_f1, test_accuracy, test_recall, test_precision, test_f1 = pm.perfclnew(inputs,best_model,train_accuracy)
    else:
        train_accuracy, train_recall,train_precision,train_f1, test_accuracy, test_recall, test_precision, test_f1, test_time = pm.perfcl(inputs,best_model, train_accuracy,train_recall,train_precision,train_f1, x_train, y_train, x_test, y_test)
    return "Random Search", train_accuracy, train_recall,train_precision,train_f1, test_accuracy,test_recall, test_precision, test_f1, best_para,time_taken,test_time, fi
  
def bayescl(inputs,x_train,y_train,x_test,y_test):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
    import perfmetric as pm
    from sklearn import svm
    import numpy as np
    import mord as m
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.ensemble import GradientBoostingClassifier
    import xgboost as xgb
    import time

    
    if inputs.at[0,'Methods'] == "Ordinal_LR":
        y_train = np.array(y_train, dtype=np.float64)
        y_train = y_train.astype(np.int)
        y_test = np.array(y_test, dtype=np.float64)
        y_test = y_test.astype(np.int)

    j = int(inputs.at[0,'CVFolds'])
    mod = inputs.at[0,'Methods']
    
    def objective(params):
        if mod == "RandomForest":
            params['n_estimators'] = int(params['n_estimators'])
            params['max_depth'] = int(params['max_depth'])
            params['min_samples_leaf'] = int(params['min_samples_leaf'])
            model = RandomForestClassifier(**params)
        if mod == "SVM":
            model = svm.SVC(**params)
        if mod == "Default_LR":
            model = LogisticRegression(solver = 'liblinear')
        if mod == "Ordinal_LR":
            model = m.LogisticIT()
        if mod == "GradientBoosting":
            params['n_estimators'] = int(params['n_estimators'])
            params['min_samples_split'] = int(params['min_samples_split'])
            params['max_depth'] = int(params['max_depth'])
            model = GradientBoostingClassifier(**params)
        if mod == "AdaBoost":
            params['n_estimators'] = int(params['n_estimators'])
            model = AdaBoostClassifier(**params)
        if mod == "XgBoost":
            params['n_estimators'] = int(params['n_estimators'])
            params['max_depth'] = int(params['max_depth'])
            model = xgb.XGBClassifier(**params)
        accuracy = cross_val_score(model, x_train, y_train, scoring = 'accuracy', cv=j).mean()
        loss = 1- accuracy            
        return {'loss': loss, 'params': params, 'status': STATUS_OK}
    
    if mod == "RandomForest":
        bmd = inputs['rf_b_md_cl'].values[0].split(",")
        bmd = list(map(int, bmd)) 
        bmsl = inputs['rf_b_msl_cl'].values[0].split(",")
        bmsl = list(map(int, bmsl)) 
        bne = inputs['rf_b_ne_cl'].values[0].split(",")
        bne = list(map(int, bne)) 
        space = {'max_depth': hp.quniform('max_depth', bmd[0],bmd[1],bmd[2]),
                 'min_samples_leaf': hp.quniform('min_samples_leaf', bmsl[0],bmsl[1],bmsl[2]),
                 'n_estimators': hp.quniform('n_estimators',bne[0],bne[1],bne[2] ),
                } 
    if mod == "AdaBoost": 
        bne = inputs['ada_b_ne_cl'].values[0].split(",")
        bne = list(map(int, bne)) 
        blr = inputs['ada_b_lr_cl'].values[0].split(",")
        blr = list(map(float, blr))
        algorithm = inputs['ada_b_algorithm_cl'].values[0].split(",")
        algorithm = list(map(str, algorithm)) 
        space = {'learning_rate': hp.loguniform('learning_rate', np.log(blr[0]),np.log(blr[1])),
                 'n_estimators': hp.quniform('n_estimators',bne[0],bne[1],bne[2] ),
                 'algorithm': hp.choice('algorithm', algorithm)
                } 
        
    if mod == "SVM":
        kernel = inputs['svm_b_kernel_cl'].values[0].split(",")
        kernel = list(map(str, kernel)) 
        c = inputs['svm_b_c_cl'].values[0].split(",")
        c = list(map(float, c)) 
        space = {'C': hp.loguniform('C', np.log(c[0]),np.log(c[1])),
                 'kernel': hp.choice('kernel', kernel)
                }
        
    if mod == "Default_LR":
        penalty = inputs['lr_b_penalty_cl'].values[0].split(",")
        penalty = list(map(str, penalty)) 
        c = inputs['lr_b_c_cl'].values[0].split(",")
        c = list(map(float, c)) 
        solver = ['liblinear']
        space = {'C': hp.loguniform('C', np.log(c[0]),np.log(c[1])),
                 'penalty': hp.choice('penalty', penalty),
                 'solver': hp.choice('solver', solver)
                }
    if mod == "Ordinal_LR":
        alpha = inputs['ord_b_alpha_cl'].values[0].split(",")
        alpha = list(map(float, alpha)) 
        space = {'alpha': hp.loguniform('alpha', np.log(alpha[0]),np.log(alpha[1])),
                }
    if mod == "GradientBoosting":
        blr = inputs['gb_b_lr_cl'].values[0].split(",")
        blr = list(map(float, blr))
        bmd = inputs['gb_b_md_cl'].values[0].split(",")
        bmd = list(map(int, bmd)) 
        bne = inputs['gb_b_ne_cl'].values[0].split(",")
        bne = list(map(int, bne)) 
        bmss = inputs['gb_b_mss_cl'].values[0].split(",")
        bmss = list(map(int, bmss)) 
        space = {'learning_rate': hp.loguniform('learning_rate', np.log(blr[0]),np.log(blr[1])),
                 'max_depth': hp.quniform('max_depth', bmd[0],bmd[1],bmd[2]),
                 'n_estimators': hp.quniform('n_estimators',bne[0],bne[1],bne[2] ),
                 'min_samples_split': hp.quniform('min_samples_split', bmss[0],bmss[1],bmss[2])
                }
    if mod == "XgBoost":
        blr = inputs['xgb_b_lr_cl'].values[0].split(",")
        blr = list(map(float, blr))
        bne = inputs['xgb_b_ne_cl'].values[0].split(",")
        bne = list(map(int, bne)) 
        bmd = inputs['xgb_b_md_cl'].values[0].split(",")
        bmd = list(map(int, bmd)) 
        space = {'learning_rate': hp.loguniform('learning_rate', np.log(blr[0]),np.log(blr[1])),
                 'n_estimators': hp.quniform('n_estimators',bne[0],bne[1],bne[2] ),
                 'max_depth': hp.quniform('max_depth', bmd[0],bmd[1],bmd[2]),
                } 
    iteration = int(inputs.at[0,'Iterations'])
    i = inputs.at[0,'TPE']
    if i == "Suggest":
        tpe_algorithm = tpe.suggest
        trials = Trials()
    if i == "Random":
        tpe_algorithm = tpe.rand.suggest
        trials = Trials()
        
    start = time.time()
    best = fmin(fn = objective, space = space, algo = tpe_algorithm, max_evals = iteration, trials=trials)
    end = time.time()
    tuning_time = end-start
    
    accuracy_list= trials.losses()
    train_accuracy = 1-min(accuracy_list)
    
    best_para_BO = space_eval(space, best)
    
    if mod == "RandomForest":
        model = RandomForestClassifier(
                            max_depth=int(best_para_BO['max_depth']),
                            n_estimators=int(best_para_BO['n_estimators']),
                            min_samples_leaf = int(best_para_BO['min_samples_leaf'])
                            )
    if mod == "SVM":
        model = svm.SVC(
                        kernel=best_para_BO['kernel'],
                        C=best_para_BO['C'],
                        )
    if mod == "Default_LR":
        model = LogisticRegression(
                        penalty= best_para_BO['penalty'],
                        C= best_para_BO['C'],
                        solver = best_para_BO['solver']
                        )
    if mod == "Ordinal_LR":
        model = m.LogisticIT(
                        alpha= best_para_BO['alpha'],
                        )
    if mod == "GradientBoosting":
        model = GradientBoostingClassifier(
                            max_depth=int(best_para_BO['max_depth']),
                            n_estimators=int(best_para_BO['n_estimators']),
                            learning_rate=best_para_BO['learning_rate'],
                            min_samples_split =int(best_para_BO['min_samples_split'])
                            ) 
    if mod == "AdaBoost":
        model = AdaBoostClassifier(
                            n_estimators=int(best_para_BO['n_estimators']),
                            learning_rate=best_para_BO['learning_rate'],
                            algorithm = best_para_BO['algorithm']
                            ) 
    if mod == "XgBoost":
        model =model = xgb.XGBClassifier(
                            n_estimators=int(best_para_BO['n_estimators']),
                            learning_rate=best_para_BO['learning_rate'],
                            max_depth=int(best_para_BO['max_depth']),
                            ) 
    model.fit(x_train,y_train)

    
    train_aacuracy, train_recall, train_precision, train_f1, test_accuracy,test_recall, test_precision, test_f1, test_time = pm.perfreg(inputs,model, train_accuracy, 0, 0, 0, x_train, y_train, x_test, y_test)
    
        
    return "Bayesian Search", mod, train_aacuracy, train_recall, train_precision, train_f1, test_accuracy,test_recall, test_precision, test_f1, best_para_BO, tuning_time, test_time, "None"

