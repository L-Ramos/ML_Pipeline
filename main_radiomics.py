
"""
Created on Mon Dec 10 15:57:11 2018

#TODO:          
        CREATE ANOTHER FILE FOR THE FEATURE EnGINEERING 
        run some Pca to check results
        try folds so I can report confusion matrix
        aspects, nihss, maybe devide into groups? or non linear relationship
        Add LASSO and XGB
        Create NN in tensorflow and optimization pipeline
        
        
        

@author: laramos
"""

import warnings
warnings.filterwarnings("ignore")
#from fancyimpute import IterativeImputer, KNN
import pandas as pd
import numpy as np
#from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
#import seaborn as sns
import pickle

import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
#from imblearn.under_sampling import RandomUnderSampler
from sklearn.compose import ColumnTransformer
import time
from sklearn.metrics import roc_auc_score
import Methods as mt
#import Methods_regression as mt_r
import utils as ut

import xgboost as xgb
import shap
import pickle


cwd = os.getcwd()
os.chdir(cwd)


    

#folder_path = r"/home/ubuntu"
folder_path = r"\\amc.intra\users\L\laramos\home\Desktop\Combination_MrCLean\data"

data_name = "complete_data_part2.csv"

data_info = "Baseline_contscore_new_review.csv"

label = 'posttici_c' #posttici_c

frame_clin, cols_orig, mask_cont, mask_cat, vals_mask, var = ut.load_clean_clinical_data(folder_path,data_name,data_info,label)
     
print("Clinical Data Loaded and Cleaned")

#frame_img = pd.read_csv(os.path.join(folder_path,"image_features_complete.csv"))
img_name = "image_features_sub_and_cort_name.csv"

frame_img,mask_cont_merge,cols_img = ut.load_clean_image_data(folder_path,img_name,label,mask_cont)
    
print("Image Data Loaded and Cleaned")

#clean the clinical data now so we can use proper train and test splits

frame_clin, Y, cols_orig, min_vals,max_vals = ut.remove_failed_img_from_clin(frame_clin,frame_img,label)


#Defining parameters for training
und='W'
opt='roc_auc'
splits=5
cv=5
mean_tprr = 0.0

     

#for imp in imputation:
imp='RF'
#for knn imputation
#clin_no_scores_with_autofeats,clinical_no_img_scores
n_neighbors=5
data_options = ['clinical','both','image','clin_no_scores_with_autofeats','clinical_no_img_scores']
#data_options = ['both','image','image','clinical_no_img_scores','clin_no_scores_with_autofeats','clinical_no_img_scores']
#data_options = ['clin_no_scores_with_autofeats','clinical_no_img_scores']
   
    
#regression or classification
task = 'classification'
#task = 'regression'

path_all_imp_data = r'\\amc.intra\users\L\laramos\home\Desktop\Combination_MrCLean\data\imputed' 

for data_to_use in data_options:

    rfc_m,svm_m,lr_m,xgb_m,nn_m = ut.create_measures(splits)     

    path_results,path_results_main,path_imp_data,need_imp,frame_img_fixed = ut.fix_create_paths(task,imp,opt,und,data_to_use,frame_img)
                     
    start_pipeline = time.time()
                
    sk = KFold(n_splits=splits, shuffle=True, random_state=1)                
    
    for l, (train_index,test_index) in enumerate(sk.split(frame_clin, Y)):
                        
        if need_imp and path_all_imp_data!='':
            X_train, X_test = frame_clin[train_index,:], frame_clin[test_index,:]
            y_train, y_test = Y_mrs[train_index], Y_mrs[test_index] 
            subj = np.array(subj)
            subj_train,subj_test = subj[train_index], subj[test_index]
            
            print("Imputing data! Iteration = ",l) 
                
            if imp=='MICE':
                X_train_imp,y_train,X_test_imp,y_test=ut.Impute_Data_MICE(X_train,y_train,X_test,y_test,1,vals_mask,cols_o,True,l) 
            else:                                                                            
                X_train_imp,y_train,X_test_imp,y_test,y_train_orig,y_test_orig=ut.Impute_Data(X_train,y_train,X_test,y_test,n_neighbors,imp,cat_vars_pos,min_vals,max_vals,var)
                                       
                save_imputed_data(path_imp_data,X_train_imp,X_test_imp,y_train,y_test,l,subj_train,subj_test)                                  
        else:
            print('Found imputation files! Loading.') 
            
        f_train,f_test,y_train,y_test = ut.load_imputed_merge_scale(path_all_imp_data,l,label,frame_img_fixed,vals_mask,mask_cont,mask_cat,mask_cont_merge,cols_img,data_to_use,task,var)
        
        #save_used_cols = pd.DataFrame(f_train.columns,columns=['name'])
        #save_used_cols.to_csv(os.path.join(path_results,'columns_used.csv'))
        
        # if 'clinical' in data_to_use or 'clin' in data_to_use or'both' in data_to_use:
        #     #removing NIHSS and other things koos idea
        #     #f_train = f_train.drop(['NIHSS_BL'],axis=1)
        #     f_train = f_train.drop(['togroin'],axis=1)
        #     #f_train = f_train.drop(['trombo'],axis=1)
        #     #f_test = f_test.drop(['NIHSS_BL'],axis=1)
        #     f_test = f_test.drop(['togroin'],axis=1)
        #     #f_test = f_test.drop(['trombo'],axis=1)
        
        #if und=='Y':
            #rus = RandomUnderSampler(random_state=1)
            #X_train_imp, y_train = rus.fit_resample(X_train_imp, y_train) 

        final_cols = f_train.columns
        save_used_cols = pd.DataFrame(final_cols,columns=['name'])
        save_used_cols.to_csv(os.path.join(path_results,'columns_used.csv'))
        break
        if task=='classification': 

            class_rfc=mt.Pipeline(True,'RFC',f_train,y_train,f_test,y_test,l,cv,mean_tprr,rfc_m,path_results,opt,und,final_cols)   
            class_svm=mt.Pipeline(True,'SVM',f_train,y_train,f_test,y_test,l,cv,mean_tprr,svm_m,path_results,opt,und,final_cols)   
            class_lr=mt.Pipeline(True,'LR',f_train,y_train,f_test,y_test,l,cv,mean_tprr,lr_m,path_results,opt,und,final_cols)
            class_nn=mt.Pipeline(True,'NN',f_train,y_train,f_test,y_test,l,cv,mean_tprr,nn_m,path_results,opt,und,final_cols) 
            class_xgb=mt.Pipeline(True,'XGB',f_train,y_train,f_test,y_test,l,cv,mean_tprr,xgb_m,path_results,opt,und,final_cols)  
        else:
            
            class_rfc = mt_r.Pipeline(True,'RFC',f_train,y_train,f_test,y_test,l,cv,mean_tprr,rfc_m,path_results,opt,und,final_cols)   
            class_svm = mt_r.Pipeline(True,'SVM',f_train,y_train,f_test,y_test,l,cv,mean_tprr,svm_m,path_results,opt,und,final_cols)   
            class_lr = mt_r.Pipeline(True,'LR',f_train,y_train,f_test,y_test,l,cv,mean_tprr,lr_m,path_results,opt,und,final_cols)
            class_nn = mt_r.Pipeline(True,'NN',f_train,y_train,f_test,y_test,l,cv,mean_tprr,nn_m,path_results,opt,und,final_cols) 
            class_xgb = mt_r.Pipeline(True,'XGB',f_train,y_train,f_test,y_test,l,cv,mean_tprr,xgb_m,path_results,opt,und,final_cols) 
        
        end_pipeline = time.time()
        print("Total time to process iteration: ",end_pipeline - start_pipeline)
                                                       
    final_m=[rfc_m,svm_m,lr_m,xgb_m,nn_m]
    final_m=[x for x in final_m if x.run != False]
    names=[class_rfc.name,class_svm.name,class_lr.name,class_xgb.name,class_nn.name]
    names=[x for x in names if x != 'NONE'] 
    if task=='classification':
        mt.Print_Results_Excel(final_m,splits,names,path_results,l,data_to_use)
    else:
        mt_r.Print_Results_Excel(final_m,splits,names,path_results,l,data_to_use)
            
    ut.Save_fpr_tpr(path_results,names,final_m)
    


