# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 10:22:42 2020

@author: laramos
"""
import re
#from missingpy import KNNImputer,MissForest
import numpy as np
import pandas as pd
import os
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer

class Measures:       
    def __init__(self,splits):
        self.auc = np.zeros(splits)
        self.brier = np.zeros(splits)
                        
        self.f1_score = np.zeros(splits)
        self.sens = np.zeros(splits)
        self.spec = np.zeros(splits)
        self.ppv = np.zeros(splits)
        self.npv = np.zeros(splits)
        
        self.fp = np.zeros(splits)
        self.fn = np.zeros(splits)
        self.tp = np.zeros(splits)
        self.tn = np.zeros(splits)
        
        # self.clf_r2=np.zeros(splits)
        # self.clf_mae=np.zeros(splits) 
        # self.clf_mse=np.zeros(splits) 
        # self.clf_mdae=np.zeros(splits)
        
        # self.clf_r_mae=np.zeros(splits) 
        # self.clf_r_mse=np.zeros(splits) 
        # self.clf_r_mdae=np.zeros(splits)
        
        self.auc_prc = np.zeros(splits)
        self.mcc = np.zeros(splits)
        self.bacc = np.zeros(splits)
        
        self.clf_tpr = list()
        self.clf_fpr = list()
        self.mean_tpr = 0.0
        self.run = False
        self.feat_imp = list() 
        self.probas = list()
        self.labels = list()
        self.shap_values = list()
        self.test_sets = list()


def Save_fpr_tpr(path_results,names,measures):
    for i in range(0,len(names)): 
        for k in range(0,len(measures[i].clf_fpr)):
            f=np.array(measures[i].clf_fpr[k],dtype='float32')
            t=np.array(measures[i].clf_tpr[k],dtype='float32')
            save_f=path_results+'fpr_'+names[i]+'_'+str(k)
            np.save(save_f,f)
            save_t=path_results+'tpr_'+names[i]+'_'+str(k)
            np.save(save_t,t)  
 
      
def Change_One_Hot(X_train_imp,X_test_imp,vals_mask):
    """
    This function one-hot-encode the features from the vals_mask and returns it as numpy array
    Input:
        frame: original frame with variables
        vals_mask: array of string with the names of the features to be one-hot-encoded [['age','sex']]
    Ouput:
        Result: One-hot-encoded feature set in pd.frame format
    """
    size = X_train_imp.shape[0]
    framen=pd.DataFrame(np.concatenate((X_train_imp,X_test_imp),axis=0),columns=X_train_imp.columns)
    framen_dummies=pd.get_dummies(framen, columns=vals_mask)    
    X_data=np.array(framen_dummies)    
    X_train_imp=(X_data[0:size,:])            
    X_test_imp=(X_data[size:,:])
    cols=framen_dummies.columns
    return(X_train_imp,X_test_imp,cols)


      
def Change_One_Hot_DL(X_train_imp,X_val_imp,X_test_imp,vals_mask):
    """
    This function one-hot-encode the features from the vals_mask and returns it as numpy array
    Input:
        frame: original frame with variables
        vals_mask: array of string with the names of the features to be one-hot-encoded [['age','sex']]
    Ouput:
        Result: One-hot-encoded feature set in pd.frame format
    """
    size_train = X_train_imp.shape[0]
    size_val = X_val_imp.shape[0]
    size_test = X_test_imp.shape[0]
    
    framen=pd.DataFrame(np.concatenate((X_train_imp,
                                        X_val_imp,
                                        X_test_imp),axis=0),columns=X_train_imp.columns)
    
    framen_dummies=pd.get_dummies(framen, columns=vals_mask)    
    X_data=np.array(framen_dummies)    
    X_train_imp=(X_data[0:size_train,:])            
    X_val_imp=(X_data[size_train:(size_train+size_val),:])
    X_test_imp=(X_data[(size_train+size_val):(size_train+size_val+size_test),:])
    cols=framen_dummies.columns

    return(X_train_imp,X_val_imp,X_test_imp,cols)
    
       
    