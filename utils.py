# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 10:22:42 2020

@author: laramos
"""
import re
from missingpy import KNNImputer,MissForest
import numpy as np
import pandas as pd
import os
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer

class Measures:       
    def __init__(self,splits):
        self.clf_auc=np.zeros(splits)
        self.clf_brier=np.zeros(splits)
                        
        self.clf_f1_score=np.zeros(splits)
        self.clf_sens=np.zeros(splits)
        self.clf_spec=np.zeros(splits)
        self.clf_ppv=np.zeros(splits)
        self.clf_npv=np.zeros(splits)
        
        self.clf_fp=np.zeros(splits)
        self.clf_fn=np.zeros(splits)
        self.clf_tp=np.zeros(splits)
        self.clf_tn=np.zeros(splits)
        
        self.clf_r2=np.zeros(splits)
        self.clf_mae=np.zeros(splits) 
        self.clf_mse=np.zeros(splits) 
        self.clf_mdae=np.zeros(splits)
        
        self.clf_r_mae=np.zeros(splits) 
        self.clf_r_mse=np.zeros(splits) 
        self.clf_r_mdae=np.zeros(splits)
        
        self.auc_prc = np.zeros(splits)
        self.mcc = np.zeros(splits)
        self.bacc = np.zeros(splits)
        
        self.clf_tpr=list()
        self.clf_fpr=list()
        self.mean_tpr=0.0
        self.run=False
        self.feat_imp=list() 
        self.probas=list()
        self.preds=list()
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
    
       
    
def Clean_Data(path_data,path_variables):


    frame=pd.read_csv(path_data,sep=';',encoding = "latin",na_values=' ')    
    for var in frame:
        if 'Study' in var:
             frame = frame.rename(columns={var: "StudySubjectID"})
    subj = frame['StudySubjectID']
    Y_mrs = frame['mrs']

    Y_mrs=np.array(Y_mrs,dtype='float32')
    
    Y_tici=frame['posttici_c'].values
    Y_tici=np.array(frame['posttici_c'].factorize(['0','1','2A','2B','2C','3'])[0],dtype="float32")
    #cnonr=frame['cnonr']
    
    miss_mrs=Y_mrs<0
    Y_mrs[miss_mrs]=np.nan
    miss_tici=Y_tici<0
    Y_tici[miss_tici]=np.nan
    
    var=pd.read_csv(path_variables)
    
    mask_cont = var[var['type']=='cont']['names']
    mask_cat = var[var['type']=='cat']['names']
    
    var=var.dropna(axis=0)

    frame=frame[var['names']]   
   
    #These are categorical with multiple categories
    vals_mask=['occlsegment_c_cbs','cbs_occlsegment_recoded','ct_bl_leukd']  
    
    cols=frame.columns
    
    data=np.zeros((frame.shape))
    
    #this features have commas instead of points for number, ruins the conversion to float
    frame['glucose']=frame['glucose'].apply(lambda x: str(x).replace(',','.'))
    frame['INR']=frame['INR'].apply(lambda x: str(x).replace(',','.'))
    frame['crpstring']=frame['crpstring'].apply(lambda x: str(x).replace(',','.'))
    frame['age']=frame['age'].apply(lambda x: str(x).replace(',','.'))
    
    #smoking =2 is missing/   prev_str =2 is missing    ivtrom =2 is missing               

    #frame['ivtrom']=frame['ivtrom'].replace(9,np.nan)
    frame['ivtci']=frame['ivtci'].replace(9,np.nan)
    frame['inhosp']=frame['inhosp'].replace(9,np.nan)
    frame['smoking']=frame['smoking'].replace(2,np.nan)
    frame['prev_str']=frame['prev_str'].replace(2,np.nan)
    frame['NIHSS_BL']=frame['NIHSS_BL'].replace(-1,np.nan)
    frame['ASPECTS_BL']=frame['ASPECTS_BL'].replace(-1,np.nan)

    for i in range(0,frame.shape[1]):
        #if frame.cols[i].dtype.name=='category':
        if var.iloc[i]['type']=='cat':
            frame[cols[i]]=frame[cols[i]].astype('category')           
            cat=frame[cols[i]].cat.categories
            frame[cols[i]],l=frame[cols[i]].factorize([np.nan,cat])
            data[:,i]=np.array(frame[cols[i]],dtype="float32")
            data[data[:,i]==-1,i]=np.nan  
        else:
            data[:,i]=np.array(frame[cols[i]],dtype="float32")
            data[data[:,i]==-1,i]=np.nan     
    

    miss=np.zeros(data.shape[1])
    for i in range(data.shape[1]):
        miss[i]=np.count_nonzero(np.isnan(data[:,i]))
    
    #return(frame,cols,var,data,Y_mrs,Y_tici,data_img)
    return(frame,cols,var,data,Y_mrs,Y_tici,vals_mask,miss,subj,mask_cont,mask_cat)
    
    
                        
    
def Impute_Data_MICE(X_train,y_train,X_test,y_test,n_imputations,vals_mask,cols,mrs, i):
    
    origin_shape = X_train.shape
    
    XY_incomplete_train = np.concatenate((X_train,y_train.reshape(-1,1)),axis=1)       
    XY_incomplete_test = np.concatenate((X_test,y_test.reshape(-1,1)),axis=1)
    
    X_train_all = np.zeros((n_imputations,X_train.shape))
    X_test_all = np.zeros((n_imputations,X_train.shape))
    y_train_all = np.zeros((n_imputations,y_train.shape))
    y_test_all = np.zeros((n_imputations,y_test.shape))
    
    for i in range(n_imputations):
        imputer = IterativeImputer(sample_posterior=True, random_state=i*10,initial_strategy="mean",min_value=0)
      
        XY_completed_train = imputer.fit_transform(XY_incomplete_train,cat_vars = np.array(cat_vars))
        XY_completed_test = imputer.transform(XY_incomplete_test)
              
        X_train_imp = (XY_completed_train[:,0:origin_shape[1]])
        y_train_imp_orig = np.array(XY_completed_train[:,origin_shape[1]],dtype="int16")
        y_train_imp = np.array(XY_completed_train[:,origin_shape[1]],dtype="int16")
        X_test_imp = (XY_completed_test[:,0:origin_shape[1]])
        y_test_imp = np.array(XY_completed_test[:,origin_shape[1]],dtype="int16")
        y_test_imp_orig = np.array(XY_completed_test[:,origin_shape[1]],dtype="int16")
    
            
        for j in range(0,X_train_imp.shape[1]):
            if  var.iloc[j]['type']=='cat':
                X_train_imp[:,j]=np.clip(np.round(X_train_imp[:,j]),min_vals[j],max_vals[j])
                X_test_imp[:,j]=np.clip(np.round(X_test_imp[:,j]),min_vals[j],max_vals[j])
            else:
                X_train_imp[:,j]=np.round(X_train_imp[:,j],decimals=0)
                X_test_imp[:,j]=np.round(X_test_imp[:,j],decimals=0)
                
        X_train_all[i,:,:] =  X_train_imp          
        X_test_all[i,:,:] =  X_test_imp          
        y_train_all[i,:] =  y_train_imp          
        y_test_all[i,:] =  y_test_imp          
        
    return(X_train_all,y_train_all,X_test_all,y_test_all)   
        
def Impute_Data(X_train,y_train,X_test,y_test,n_neighbors,imputer,cat_vars,min_vals,max_vals,var):
    
    origin_shape = X_train.shape
    
    XY_incomplete_train = np.concatenate((X_train,y_train.reshape(-1,1)),axis=1)       
    XY_incomplete_test = np.concatenate((X_test,y_test.reshape(-1,1)),axis=1)
    
    if imputer=='KNN':
        imputer = KNNImputer(n_neighbors=n_neighbors)
    else:
        imputer = MissForest(random_state=1,n_jobs=-1)
        
    XY_completed_train = imputer.fit_transform(XY_incomplete_train,cat_vars = np.array(cat_vars))
    XY_completed_test = imputer.transform(XY_incomplete_test)
          
    X_train_imp = (XY_completed_train[:,0:origin_shape[1]])
    y_train_imp_orig = np.array(XY_completed_train[:,origin_shape[1]],dtype="int16")
    y_train_imp = np.array(XY_completed_train[:,origin_shape[1]],dtype="int16")
    X_test_imp = (XY_completed_test[:,0:origin_shape[1]])
    y_test_imp = np.array(XY_completed_test[:,origin_shape[1]],dtype="int16")
    y_test_imp_orig = np.array(XY_completed_test[:,origin_shape[1]],dtype="int16")

        
    for j in range(0,X_train_imp.shape[1]):
        if  var.iloc[j]['type']=='cat':
            X_train_imp[:,j]=np.clip(np.round(X_train_imp[:,j]),min_vals[j],max_vals[j])
            X_test_imp[:,j]=np.clip(np.round(X_test_imp[:,j]),min_vals[j],max_vals[j])
        else:
            X_train_imp[:,j]=np.round(X_train_imp[:,j],decimals=0)
            X_test_imp[:,j]=np.round(X_test_imp[:,j],decimals=0)
    
    #min_vals_imp=np.nanmin(np.concatenate((X_train_imp,X_test_imp),axis=0),axis=0)
    #max_vals_imp=np.nanmax(np.concatenate((X_train_imp,X_test_imp),axis=0),axis=0)  
                    
    return(X_train_imp,y_train_imp,X_test_imp,y_test_imp,y_train_imp_orig,y_test_imp_orig)  
    
def Impute_Data_DL(X_train,X_test,X_val,n_neighbors,imputer,mask_cont,min_vals,max_vals,var):
    
    X_train = frame_clin_train
    X_val = frame_clin_val
    X_test = frame_clin_test
    #cat_vars = np.array(cont_vars_pos).reshape(-1,1)
    
    origin_shape = X_train.shape

    subj_train = X_train['ID']
    subj_val = X_val['ID']
    subj_test = X_test['ID']
    
    X_train = X_train.drop('ID',axis=1)
    X_test = X_test.drop('ID',axis=1)
    X_val = X_val.drop('ID',axis=1)
    
    orig_cols = X_train.columns
    
    XY_incomplete_train =np.array(X_train)
    XY_incomplete_val =np.array(X_val)
    XY_incomplete_test = np.array(X_test)
    
    cont_vars_pos, cat_vars_pos = get_pos_cont_and_cat_variables(mask_cont,X_train.columns)
    min_vals = np.nanmin(X_train,axis=0)
    max_vals = np.nanmax(X_train,axis=0)

    
    if imputer=='KNN':
        imputer = KNNImputer(n_neighbors=n_neighbors)
    else:
        imputer = MissForest(random_state=1,n_jobs=-1)
        
    XY_completed_train = imputer.fit_transform(XY_incomplete_train,cat_vars = np.array(cat_vars_pos))
    XY_completed_test = imputer.transform(XY_incomplete_test)
    XY_completed_val = imputer.transform(XY_incomplete_val)
          
    X_train_imp=(XY_completed_train[:,0:origin_shape[1]])
    X_test_imp=(XY_completed_test[:,0:origin_shape[1]])
    X_val_imp=(XY_completed_val[:,0:origin_shape[1]])

        
    for j in range(0,X_train_imp.shape[1]):
        #if  var.iloc[j]['type']=='cat':
        if  j in cat_vars_pos:
            X_train_imp[:,j]=np.clip(np.round(X_train_imp[:,j]),min_vals[j],max_vals[j])
            X_test_imp[:,j]=np.clip(np.round(X_test_imp[:,j]),min_vals[j],max_vals[j])
            X_val_imp[:,j]=np.clip(np.round(X_val_imp[:,j]),min_vals[j],max_vals[j])
        else:
            X_train_imp[:,j]=np.round(X_train_imp[:,j],decimals=0)
            X_test_imp[:,j]=np.round(X_test_imp[:,j],decimals=0)
            X_val_imp[:,j]=np.round(X_val_imp[:,j],decimals=0)
    
    X_train_imp = pd.DataFrame(X_train_imp,columns = orig_cols)
    X_val_imp = pd.DataFrame(X_val_imp,columns = orig_cols)
    X_test_imp = pd.DataFrame(X_test_imp,columns = orig_cols)
    X_train_imp['ID'] = subj_train
    X_val_imp['ID'] = subj_val
    X_test_imp['ID'] = subj_test
                    
    return(X_train_imp,X_val_imp,X_test_imp)  
          
def get_pos_cont_and_cat_variables(mask_cont,cols_o):
    cont_vars_pos = list()
    mask_cont = list(mask_cont)
    for i in range(0,len(mask_cont)):
            cont_vars_pos.append(np.where(mask_cont[i]==cols_o)[0][0])        
    cat_vars_pos = (list(set(np.arange(0,len(cols_o))) - set(cont_vars_pos)))
    return(cont_vars_pos,cat_vars_pos)


def get_ids_images(path):
    id_done_list=list()
    done = sorted(os.listdir(path))
    for f in done:
        id_p = re.search('R[0-9][0-9][0-9][0-9]',f)
        if id_p:
            id_done_list.append(id_p[0])  
    return(id_done_list)


def clean_mask(vals_mask,cols_o):
    vals = list()
    for v in vals_mask:
        if v in cols_o:
            vals.append(v)
    vals_mask = vals  
    return(vals_mask)




def rename_image_features(frame_img,image_columns):
    for i,val in enumerate(image_columns):
        if val!='ID':
            image_columns[i]='feat'+str(val)
    frame_img.columns = image_columns
    #image_columns.pop()
    return(frame_img)

def remove_failed_path(frame,folder_path,id_col):
    #removes the features from images that have a folder but failed (failed was defined by visual inspection)
    id_done_part1 = ut.get_ids_images(r'L:\basic\Personal Archive\L\laramos\Disk E\MrClean_part1')
    id_done_part2 = ut.get_ids_images(r'L:\basic\Personal Archive\L\laramos\Disk E\MrClean_part2')
    comb_ids = id_done_part1 +id_done_part2
    np.save(os.path.join(folder_path,'all_clin_ids'),comb_ids)
    
    f_part1 = pd.read_csv(r'\\amc.intra\users\L\laramos\home\Desktop\Combination_MrCLean\imaging/failed_part1.csv')
    f_part2 = pd.read_csv(r'\\amc.intra\users\L\laramos\home\Desktop\Combination_MrCLean\imaging/failed_part2.csv')
    f_part2 = f_part2[f_part2.status=='Failed']
    failed_ids = list(f_part1['s_ids'])+list(f_part2['s_ids'])
    np.save(os.path.join(folder_path,'failed_img_ids'),failed_ids)
            
    all_ids = [i for i in comb_ids if i not in failed_ids]    
    #frame_failed = frame[~frame.s_id.isin(all_ids)]
    frame = frame[frame[id_col].isin(all_ids)]
    return(frame)
 
def remove_failed_file(frame,folder_path,id_col):
    #removes the features from images that have a folder but failed (failed was defined by visual inspection)
 
    comb_ids = np.load(os.path.join(folder_path,'all_clin_ids.npy'))
    
    failed_ids = np.load(os.path.join(folder_path,'failed_img_ids.npy'))
            
    all_ids = [i for i in comb_ids if i not in failed_ids]    
    #frame_failed = frame[~frame.s_id.isin(all_ids)]
    frame = frame[frame[id_col].isin(all_ids)]
    return(frame)   

    
def remove_correlated(frame,label,id_col,plot=False, threshold=0.99,clin=False):
    # remove columns that are too correlate
    subj = frame[id_col]
    frame = frame.drop(id_col,axis=1)
    if clin:
        Y_mrs = np.array(frame[label])
        frame = frame.drop(label,axis=1)
    corr = frame.corr()
    mask = np.zeros_like(corr, dtype=np.bool)
    
    mask[np.triu_indices_from(mask)] = True
    # Set up the matplotlib figure
   
    # Draw the heatmap with the mask and correct aspect ratio
    if plot:
        f, ax = plt.subplots(figsize=(25, 15))
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1.0, center=0,
                    square=True, linewidths=.5, cbar_kws={"shrink": .5})
    
    corr_matrix = frame.corr().abs()
    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    # Find index of feature columns with correlation greater than 0.95
    to_drop = [column for column in upper.columns if any(upper[column] >threshold) or sum(frame[column])==0]
    
    frame=frame.drop(frame[to_drop], axis=1)
    #if clin:
    #    frame['y_mrs'] = Y_mrs
    frame[id_col] = subj
    return(frame)
    
             
def save_imputed_data(path_imp_data,X_train_imp,X_test_imp,y_train,y_test,l,subj_train,subj_test):
    X_train_imp = pd.DataFrame(X_train_imp,columns=cols_o[0:X_train_imp.shape[1]])
    X_test_imp = pd.DataFrame(X_test_imp,columns=cols_o[0:X_test_imp.shape[1]])
    
    X_train_imp['s_id'] = subj_train
    X_train_imp['mrs'] = y_train
    X_test_imp['s_id'] = subj_test
    X_test_imp['mrs'] = y_test
    
    X_train_imp.to_csv(os.path.join(path_imp_data,('imp_data_train'+str(l)+".csv")),index=False)
    X_test_imp.to_csv(os.path.join(path_imp_data,('imp_data_test'+str(l)+".csv")),index=False)
    np.save(os.path.join(path_imp_data,('imp_y_train'+str(l)+".npy")),y_train)
    np.save(os.path.join(path_imp_data,('imp_y_test'+str(l)+".npy")),y_test) 
    
def load_imputed_data(path_imp_data,l,label):   
    X_train_imp = pd.read_csv(path_imp_data+'imp_data_train'+str(l)+".csv")
    X_test_imp = pd.read_csv(path_imp_data+'imp_data_test'+str(l)+".csv")
    subj_train = X_train_imp['s_id']
    subj_test = X_test_imp['s_id']
    y_train = np.load(path_imp_data+'imp_y_train'+str(l)+".npy")
    y_test = np.load(path_imp_data+'imp_y_test'+str(l)+".npy")
    X_train_imp = X_train_imp.drop(X_train_imp[['s_id',label]], axis=1)
    X_test_imp = X_test_imp.drop(X_test_imp[['s_id',label]], axis=1)
    
    return(X_train_imp,X_test_imp,y_train,y_test,subj_train,subj_test)

def load_imputed_data_overall(path_imp_data,l,label):   
    X_train_imp = pd.read_csv(path_imp_data+'//imp_data_train'+str(l)+".csv")
    X_test_imp = pd.read_csv(path_imp_data+'//imp_data_test'+str(l)+".csv")
    subj_train = X_train_imp['s_id']
    subj_test = X_test_imp['s_id']
    y_train = pd.read_csv(path_imp_data+'//imp_y_train'+str(l)+".csv")
    y_test = pd.read_csv(path_imp_data+'//imp_y_test'+str(l)+".csv")
    #X_train_imp = X_train_imp.drop(X_train_imp[['s_id',label]], axis=1)
    #X_test_imp = X_test_imp.drop(X_test_imp[['s_id',label]], axis=1)
    y_train = np.array(y_train[label])
    y_test = np.array(y_test[label])
    
    return(X_train_imp,X_test_imp,y_train,y_test,subj_train,subj_test)
                    
def merge_clinical_image_data(subj_train,subj_test,X_train_imp,X_test_imp,y_train,y_test,frame_img,clin_vars):
    
    X_train_imp = pd.DataFrame(X_train_imp,columns=clin_vars)
    X_test_imp = pd.DataFrame(X_test_imp,columns=clin_vars)
    
    X_train_imp['s_id'] = subj_train
    X_train_imp['mrs'] = y_train
    X_test_imp['s_id'] = subj_test
    X_test_imp['mrs'] = y_test
    
    X_train_imp = X_train_imp.merge(frame_img,left_on='s_id',right_on='ID')
    X_test_imp = X_test_imp.merge(frame_img,left_on='s_id',right_on='ID')
    
    y_train = np.array(X_train_imp['mrs'])
    y_test = np.array(X_test_imp['mrs'])
    X_train_imp = X_train_imp.drop(['s_id','mrs','ID'], axis=1)
    X_test_imp = X_test_imp.drop(['s_id','mrs','ID'], axis=1)
    
    return(X_train_imp,X_test_imp,y_train,y_test)
                    
def fix_create_paths(name,imp,opt,und,data_to_use,frame_img):
    path_results=(r".//complete_radio_no_image"+name+imp+"_opt-"+opt+"_und-"+und+"//")
    path_results_main = path_results
    if not os.path.exists(path_results):
        os.makedirs(path_results)
    if 'clinical' in data_to_use:
        frame_img = frame_img['ID']
        path_results = os.path.join(path_results,data_to_use)
        if not os.path.exists(path_results):            
            os.makedirs(path_results)
    else:
        if data_to_use=='both':
            path_results = path_results+r'\combination\\'
        else:
            if data_to_use=='image':
                path_results = path_results+r'\image\\'
            else:
                if data_to_use=='clin_no_scores_with_autofeats':
                    path_results = path_results+r'\clin_no_scores_with_autofeats\\'
        if not os.path.exists(path_results):
            os.makedirs(path_results) 
            
            
    path_imp_data = path_results_main+r'data//'
    if not os.path.exists(path_imp_data):
        os.makedirs(path_imp_data)
        need_imp = True
    else:
        need_imp = False
    return(path_results,path_results_main,path_imp_data,need_imp,frame_img)

def load_clean_clinical_data(folder_path,data_name,data_info,label):
    
    path_data = os.path.join(folder_path,data_name)

    path_variables = os.path.join(folder_path,data_info)
    
    _,cols_o,var,data,Y_mrs,Y_tici,vals_mask,miss,subj,mask_cont,mask_cat=Clean_Data(path_data,path_variables)
    
    cont_vars_pos, cat_vars_pos = get_pos_cont_and_cat_variables(mask_cont,cols_o)
        
    frame_clin = pd.DataFrame(data,columns=cols_o)
    frame_clin['s_id'] = subj
    if label=='mrs':
         frame_clin[label] = Y_mrs
    else:
         frame_clin[label] = Y_tici
    
    frame_clin = remove_correlated(frame_clin,label,id_col='s_id',plot=False,threshold=0.85,clin=True)
    #frame_clin = frame_clin.drop(['s_id'], axis=1)
    
    vals_mask = clean_mask(vals_mask,cols_o)
    mask_cont = clean_mask(mask_cont,cols_o)
    mask_cat = clean_mask(mask_cat,cols_o)
    
    clin_vars = list(mask_cont) + list(mask_cat)
    
    #data = np.array(frame_clin)
    cols_orig = frame_clin.columns
    min_vals = np.nanmin(data,axis=0)
    max_vals = np.nanmax(data,axis=0)
    
    if label=='mrs':
         frame_clin[label] = Y_mrs
    else:
         frame_clin[label] = Y_tici

    return(frame_clin,cols_orig,mask_cont,mask_cat,vals_mask,var)
    

def load_clean_image_data(folder_path,img_name,label,mask_cont):
    frame_img = pd.read_csv(os.path.join(folder_path,img_name))

    frame_img = rename_image_features(frame_img,list(frame_img.columns))
    
    #threshold was 0.85 before
    frame_img = remove_correlated(frame_img,label,id_col='ID',plot=False,threshold=0.50,clin=False)
    
    frame_img = remove_failed_file(frame_img,folder_path,id_col='ID')
    
    mask_cont = list(mask_cont)
    cols_img = list(frame_img.columns)
    cols_img.pop()
    mask_cont_all = mask_cont + cols_img
    
    return frame_img,mask_cont_all, cols_img

def remove_failed_img_from_clin(frame_clin,frame_img,label):
        
    f = frame_img['ID']
    frame_clin = frame_clin.merge(f,left_on='s_id',right_on='ID')
    frame_clin = frame_clin.drop_duplicates()
    subj = frame_clin['s_id']
    Y = np.array(frame_clin[label])
    frame_clin = frame_clin.drop(['s_id',label,'ID'], axis=1)
    
   
    cols_orig = frame_clin.columns
    min_vals = np.nanmin(frame_clin,axis=0)
    max_vals = np.nanmax(frame_clin,axis=0)
    return frame_clin, Y, cols_orig, min_vals,max_vals

def create_measures(splits):
    rfc_m = Measures(splits)
    svm_m = Measures(splits)
    lr_m = Measures(splits)
    xgb_m = Measures(splits)
    nn_m = Measures(splits) 
    return rfc_m,svm_m,lr_m,xgb_m,nn_m


def load_imputed_merge_scale(path_all_imp_data,l,label,frame_img,vals_mask,mask_cont,mask_cat,mask_cont_all,cols_img,data_to_use,task,var):
    
    X_train_imp,X_test_imp,y_train,y_test,subj_train,subj_test = load_imputed_data_overall(path_all_imp_data,l,label)
    
    X_train_imp, X_test_imp,y_train,y_test = merge_clinical_image_data(subj_train,subj_test,X_train_imp,X_test_imp,y_train,y_test,frame_img,list(mask_cont) + list(mask_cat)) 
    
    X_train_imp,X_test_imp,cols_recoded = Change_One_Hot(X_train_imp,X_test_imp,vals_mask)
     
    if 'clinical' in data_to_use:
        scaler = ColumnTransformer([('norm1', StandardScaler(),mask_cont)], remainder='passthrough')
    else:
        #mask_cont_all = mask_cont + cols_img            
        scaler = ColumnTransformer([('norm1', StandardScaler(),mask_cont_all)], remainder='passthrough')
                                   
    f_train = pd.DataFrame(X_train_imp,columns=cols_recoded)
    f_test = pd.DataFrame(X_test_imp,columns=cols_recoded)
     
    # scaler = scaler.fit(f_train)    
    # X_train_imp = scaler.transform(f_train)
    # X_test_imp = scaler.transform(f_test)
    # #mask_cont_all.extend(mask_cat)
    # # if 'clinical' in data_to_use or 'clin' in data_to_use or 'both' in data_to_use:
    # f_train = pd.DataFrame(X_train_imp,columns = cols_recoded)
    # f_test = pd.DataFrame(X_test_imp,columns = cols_recoded)
    # else:
    #     f_train = pd.DataFrame(X_train_imp,columns = mask_cont_all)
    #     f_test = pd.DataFrame(X_test_imp,columns = mask_cont_all)

    if data_to_use=='image':
        f_train = f_train[cols_img]
        f_test = f_test[cols_img]
    elif data_to_use == 'clinical_no_img_scores':
        f_train = f_train[var[var['img']==0]['names']]
        f_test = f_test[var[var['img']==0]['names']]
    elif data_to_use == 'clin_no_scores_with_autofeats':
        f_train = f_train[list(var[var['img']==0]['names'])+list(cols_img)]
        f_test = f_test[list(var[var['img']==0]['names'])+list(cols_img)]
    
    if task=='classification': 
        if label=='mrs':
            y_train = y_train<=2
            y_test = y_test<=2
        else:
            y_train = y_train>=3
            y_test = y_test>=3
        
    return f_train,f_test,y_train,y_test
