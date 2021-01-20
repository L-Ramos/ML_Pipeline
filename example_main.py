# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 14:27:26 2021

@author: laramos
"""

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold
import pandas as pd
import os
import numpy as np

#import data_clean as dt
import methods as mt
import utils as ut

args = dict()
args['random_state'] = 1         
args['undersample'] = 'W'
args['n_jobs'] = -2
args['outter_splits'] = 3
args['verbose'] = 1
args['n_iter_search'] = 1
args['opt_measure'] = 'roc_auc'
args['cv_plits'] = 3
args['class_weight'] = 'balanced'

PATH_RESULTS = r"\\amc.intra\users\L\laramos\home\Desktop\Postdoc eHealth\Results"

name = 'test_cancer'

path_write = os.path.join(PATH_RESULTS,name)

if not os.path.exists(path_write):
                os.mkdir(path_write)
                               
data = load_breast_cancer()

X = pd.DataFrame(data.data,columns=data.feature_names) 
y = data.target

kf = KFold(n_splits = args['outter_splits'], random_state=1, shuffle=True)

args['mask_cont'] = X.columns
args['mask_cat'] = X.columns
            
rfc_m, lr_m, _, _, _, meas = ut.create_measures(args['outter_splits'])

for fold,(train_index, test_index) in enumerate(kf.split(X)):
               X_train,y_train = X.iloc[train_index,:], y[train_index]
               X_test,y_test = X.iloc[test_index,:], y[test_index]
               args['current_iteration'] = fold
               args['pos_weights'] = (y_train.shape[0]-sum(y_train))/sum(y_train)
               grid_rfc = mt.RandomForest_CLF(args,X_train,y_train,X_test,y_test,rfc_m,test_index)
               grid_lr = mt.LogisticRegression_CLF(args,X_train,y_train,X_test,y_test,lr_m,test_index)                             
               names = ['RFC','LR']
               mt.print_results_excel(meas,names,path_write)
               mt.plot_shap(X,args,path_write, rfc_m.feat_names)