# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 18:38:43 2020

@author: laramos
"""

class Measures:       
    def __init__(self,splits):
        
        self.auc=np.zeros(splits)                    
        self.f1_score=np.zeros(splits)
        self.sens=np.zeros(splits)
        self.spec=np.zeros(splits)
        self.ppv=np.zeros(splits)
        self.npv=np.zeros(splits)
        
        self.fp=np.zeros(splits)
        self.fn=np.zeros(splits)
        self.tp=np.zeros(splits)
        self.tn=np.zeros(splits)
        
        self.auc_prc = np.zeros(splits)
        self.mcc = np.zeros(splits)
        self.bacc = np.zeros(splits)
        
        self.clf_tpr=list()
        self.clf_fpr=list()
        self.mean_tpr=0.0
        self.run=False
        self.feat_imp=list() 
        self.probas=list()
        self.labels = list()
        self.preds=list()
        self.shap_values = list()
        self.test_sets = list()

def get_SVM():
    
        tuned_parameters = {
        'clf__C':            ([0.1, 0.01, 0.001, 1, 10, 100]),
        'clf__kernel':       ['linear', 'rbf','poly'],                
        'clf__degree':       ([1,2,3,4,5,6]),
        'clf__gamma':         [1, 0.1, 0.01, 0.001, 0.0001]
        #'tol':         [1, 0.1, 0.01, 0.001, 0.0001],
        }
        return(tuned_parameters)
    
def get_RFC():
    
        tuned_parameters = {
        'clf__n_estimators': ([200,400,500,600,800,1000,1200,1400]),
        'clf__max_features': (['auto', 'sqrt', 'log2']),                   # precomputed,'poly', 'sigmoid'
        'clf__max_depth':    ([10,20,30,40, 50, 60, 70, 80, 90, 100, None]),
        'clf__criterion':    (['gini', 'entropy']),
        'clf__min_samples_split':  [2,4,6,8],
        'clf__min_samples_leaf':   [2,4,6,8,10]}
        return(tuned_parameters)
        
def get_LR(): 
       
        tuned_parameters = {
        'clf__C': [0.001, 0.01, 0.1, 1, 10, 100],
        'clf__solver' : ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
        'clf__max_inter' : [5000]}
        return(tuned_parameters)
        
def get_NN():
    
        tuned_parameters = {
        'clf__activation': (['relu','logistic']),
        'clf__hidden_layer_sizes':([[60,120,60],[60,120,120],[60,60],[60,120]]),
        #'hidden_layer_sizes':([[131,191,131],[131,231,131],[131,131,131]]),
        'clf__alpha':     ([0.01, 0.001, 0.0001]),
        'clf__batch_size':         [32,64],
        'clf__learning_rate_init':    [0.01, 0.001],
        'clf__solver': ["adam"]}
        return(tuned_parameters)
        
#def get_LASSO():
    
def get_XGB():
    
        tuned_parameters = {
        'clf__learning_rate': ([0.1, 0.01, 0.001]),
        #'gamma': ([100,10,1, 0.1, 0.01, 0.001]),                  
        #'max_depth':    ([3,5,10,15]),
        #'subsample ':    ([0.5,1]),
        #'reg_lambda ':  [1,10,100],
        #'alpha ':   [1,10,100],
        
        'clf__min_child_weight': [1, 5, 10],
        'clf__gamma': [0, 0.5, 1, 1.5, 2, 5],
        'clf__subsample': [0.7, 0.8, 0.9, 1.0],
        'clf__colsample_bytree': [0.3,0.4,0.5,0.6,0.7,0.8],
        'clf__max_depth': [3, 5, 7, 9, 10]}
        return(tuned_parameters)