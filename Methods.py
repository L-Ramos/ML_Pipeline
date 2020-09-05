"""
This files contains the classes and functions for creating the machine learning methods
"""



import numpy as np
from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,precision_recall_curve,matthews_corrcoef,balanced_accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV                               
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import brier_score_loss,f1_score
import random as rand
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import warnings
import time
from scipy.stats import randint as sp_randint
import scipy as sp
warnings.filterwarnings("ignore", category=DeprecationWarning)
from scipy import interp 
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import RFECV
from sklearn.naive_bayes import GaussianNB
import xlwt
from sklearn.metrics import make_scorer
import xgboost as xgb
from sklearn.utils.class_weight import compute_sample_weight
import joblib
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import shap
import pickle
from sklearn.pipeline import Pipeline
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


n_jobs=-2

random_state=1

import hyperparameters as hp





class ClassifierPipeline():
    """Define main functions for classifiers."""
    def __init__ (self,args):
        self.random_state = args['random_state']
        self.und = args['undersample']
        self.n_jobs = args['n_jobs']
        self.n_iter = args['n_iter_search']
        self.verbose = args['verbose']
        self.opt = args['opt_measure']
        self.splits = args['cv_plits']
        self.data_to_use = args['data_to_use']
        self.mask_cont = args['mask_cont']
        self.mask_cont_merge = args['mask_cont_merge']
        self.iter = args['current_iteration']
        self.imp = None
        self.pos_weights = args['pos_weight']
        self.class_weight = args['class_weight']
        

    def get_scaler(self):
        """Create the scaler based on columns that need to be standardized."""        
        if 'clinical' in self.data_to_use:
            scaler = ColumnTransformer([('norm1', StandardScaler(),self.mask_cont)], remainder='passthrough')
        else:           
            scaler = ColumnTransformer([('norm1', StandardScaler(),self.mask_cont_merge)], remainder='passthrough')
        return scaler    
    
    def get_imputer(self):
        """Create the imputer for missing data."""
        imp = IterativeImputer(random_state=self.random_state)
        return imp
        
    def def_pipeline(self):
        """Defines the pipeline with all the operations to be run.""" 
              
        if self.imp!=None:
            pipe = Pipeline([('scaler', self.scaler),('imputer', self.imp), ('clf', self.clf)])
        else:
            pipe = Pipeline([('scaler', self.scaler), ('clf', self.clf)])
        return pipe
    
    def run_grid_search(self,x_train,y_train,x_test,y_test):
        """Perform random grid seach to optimize hyperparameters.
        
        Args:
            x_train (DataFrame): Training set.
            y_train (array): Label for the training set
            x_test (DataFrame): Testing set.
            y_test (array): Label for the testing set

        Returns:
            preds (array): binary predictions for the test set
            probas (array):probability predictions for the test set
        """  
                                         
        grid =  RandomizedSearchCV(self.pipe, self.params, cv=self.splits,random_state=self.random_state,
                           scoring='%s' % self.opt,n_jobs=self.n_jobs,n_iter=self.n_iter,verbose=self.verbose)  

        grid = grid.fit(x_train,y_train)                                                
        #self.best_params = clf_grid.best_params_
        #self.best_clf = grid.best_estimator_
        preds = grid.predict(x_test)
        probas = grid.predict_proba(x_test)[:,1]
        return preds, probas
    
    def evaluation_measures(self,y_test,meas):
        """Perform random grid seach to optimize hyperparameters.
        
        Args:
            y_test (array): testing labels
            meas (object class Measures): instance from class that stores measures over iterations
            
        Returns:
            meas (object class Measures): updated instance
        
        """
        precision, recall, _ = precision_recall_curve(y_test,self.probas)
        meas.f1_score[self.iter] = f1_score(y_test, self.preds)
        meas.auc[self.iter] = roc_auc_score(y_test,self.probas)
        
        tn, fp, fn, tp = confusion_matrix(y_test, self.preds).ravel()  
        
        meas.sens[self.iter] = tp/(tp+fn)
        meas.spec[self.iter] = tn/(tn+fp)
        meas.ppv[self.iter] = tp/(tp+fp)
        meas.npv[self.iter] = tn/(tn+fn)                                
        meas.auc_prc[self.iter] = (auc(recall,precision))
        meas.mcc[self.iter] = (matthews_corrcoef(y_test,self.preds))
        meas.bacc[self.iter] = (balanced_accuracy_score(y_test,self.preds)) 
        
        meas.probas.append(self.probas)
        meas.labels.append(y_test)
        
        return meas
        
    
class InitPipeline(ClassifierPipeline):
    """ Initialize and run the pipeline for all classifiers. """
    def __init__ (self,args,x_train,y_train,x_test,y_test,meas):
        """ Run pipeline.
        
        Args:
            x_train (DataFrame): Training set.
            y_train (array): Label for the training set
            x_test (DataFrame): Testing set.
            y_test (array): Label for the testing set
            meas (object class Measures): instance from class that stores measures over iterations

        Returns:
            meas (object class Measures): updated instance
        """  
        
        self.params = self.get_parameter_grid()
        ClassifierPipeline.__init__(self,args)
        self.clf = self.define_classifier()   
        self.scaler = self.get_scaler()  
        self.imp = self.get_imputer()
        self.pipe = self.def_pipeline()
        self.preds,self.probas = self.run_grid_search(x_train,y_train,x_test,y_test)        
        meas, self.evaluation_measures(y_test,meas)
        return meas
    

class RandomForest_CLF(InitPipeline):
    """ Initialize and run the Random Forest Classifier. """
    
    def get_parameter_grid(self):
        """ Read hyper-parameters for the Random Forest Classifier. """
        return hp.get_RFC()
    
    def define_classifier(self):
        """ Instantiate the classifier based or not on class weights. """
        if self.und=='W':
            clf = RandomForestClassifier(n_estimators=25, oob_score = True,random_state=self.random_state,class_weight=self.class_weight)   
        else:
            clf = RandomForestClassifier(n_estimators=25, oob_score = True,random_state=self.random_state)  
        return(clf)
    
    def __init__ (self,args,x_train,y_train,x_test,y_test,meas):
        
        meas = InitPipeline.__init__(self,args,x_train,y_train,x_test,y_test,meas)



class LogisticRegression_CLF(InitPipeline):
    
    def get_parameter_grid(self):
        return hp.get_RFC()
    
    def define_classifier(self):
        if self.und=='W':
            clf = LogisticRegression(random_state=self.random_state,max_iter=self.params['clf__max_inter'],class_weight=self.class_weight)         
        else:
            clf = LogisticRegression(random_state=self.random_state,max_iter=self.params['clf__max_inter']) 
        return(clf)
    
    def __init__ (self,args,x_train,y_train,x_test,y_test,meas):

        meas = InitPipeline.__init__(self,args,x_train,y_train,x_test,y_test,meas)
   
        
class SupportVectorMachine_CLF(InitPipeline):
    
    def get_parameter_grid(self):
        return hp.get_SVM()
    
    def define_classifier(self):
        if self.und=='W':
            clf = SVC(random_state=random_state,class_weight=self.class_weight,probability=True)            
        else:
            clf = SVC(random_state=random_state,probability=True)            
        return(clf)
    
    def __init__ (self,args,x_train,y_train,x_test,y_test,meas):

        meas = InitPipeline.__init__(self,args,x_train,y_train,x_test,y_test,meas)        
        
        
class XGBoost_CLF(InitPipeline):
    
    def get_parameter_grid(self):
        return hp.get_XGB()
    
    def define_classifier(self):
        if und=='W':
            clf = xgb.XGBClassifier(random_state=self.random_state,scale_pos_weight=self.pos_weight) 
        else:                                    
            clf = xgb.XGBClassifier(random_state=random_state)            
        return(clf)
    
    def __init__ (self,args,x_train,y_train,x_test,y_test,meas):

       meas =  InitPipeline.__init__(self,args,x_train,y_train,x_test,y_test,meas)       
        
        
        
class MultilayerPerceptron_CLF(InitPipeline):
    
    def get_parameter_grid(self):
        return hp.get_XGB()
    
    def define_classifier(self):        
            clf = MLPClassifier(hidden_layer_sizes=(x_train.shape[1]),max_iter=self.,batch_size=32,random_state=self.random_state )
        return(clf)
    
    def __init__ (self,args,x_train,y_train,x_test,y_test,meas):

        meas = InitPipeline.__init__(self,args,x_train,y_train,x_test,y_test,meas)        
                
        

 if name == 'NN':
    print("NN Grid Search")
    tuned_parameters = get_NN()                                                        
    clf =                          
            
else:                    
    if name == 'XGB':
        print("XGB Grid Search")
        tuned_parameters = get_XGB()
        if und=='W':
            clf = xgb.XGBClassifier(random_state=random_state,scale_pos_weight=(y_train.shape[0]-sum(y_train))/sum(y_train)) 
        else:                                    
            clf = xgb.XGBClassifier(random_state=random_state)        
        
                
args = dict()
args['random_state'] = 1        
args['undersample'] = 'W'
args['n_jobs'] = -2
args['n_iter_search'] = 5
args['verbose'] = 1
args['opt_measure'] = 'roc_auc'
args['cv_plits'] = 5
args['data_to_use'] = data_to_use
args['mask_cont'] = mask_cont
args['mask_cont_merge'] = mask_cont_all
args['current_iteration'] = fold
args['class_weight'] = 'balanced'
args['pos_weights'] = (y_train.shape[0]-sum(y_train))/sum(y_train)

meas = Measures(splits)
c = RandomForest(args,f_train,y_train,f_test,y_test,meas)
c = LogisticRegression_CLF(args,f_train,y_train,f_test,y_test,meas)
c = SupportVectorMachine_CLF(args,f_train,y_train,f_test,y_test,meas)


def shape_visualization(x_test, m, clf,name):

        #testing feature importance with xgb
        #x_train_t = pd.DataFrame(x_train,columns=new_cols)
        #f = plt.figure(figsize=(25, 19))
        #xgboost.plot_importance(clf_t,importance_type="gain")

        if name=='LR':
            explainer = shap.LinearExplainer(clf, x_test,feature_perturbation='correlation_dependent',check_additivity=True)
            shap_values = explainer.shap_values(x_test)
            m.shap_values.append(shap_values)
            m.test_sets.append(x_test)
        elif name=='RFC':
            explainer = shap.TreeExplainer(clf)        
            shap_values = explainer.shap_values(x_test)
            m.shap_values.append(shap_values)
            m.test_sets.append(x_test)
        
        #shap.summary_plot(shap_values, x_train, plot_type="bar")
        #shap.summary_plot(shap_values, x_train)
#        #end of test


def plot_shap():
    
    folder = r"\\amc.intra\users\L\laramos\home\Desktop\Combination_MrCLean\complete_radio_no_imageclassificationRF_opt-roc_auc_und-W/tici"
    #['clinical','combination','image','clin_no_scores_with_autofeats','clinical_no_img_scores']
    type_var = "combination"
    path = os.path.join(folder,type_var)
    
    #file = r"\\amc.intra\users\L\laramos\home\Desktop\Combination_MrCLean\complete_radio_no_imageclassificationRF_opt-roc_auc_und-W\image\measures_LR.pkl"
    #file = r"\\amc.intra\users\L\laramos\home\Desktop\Combination_MrCLean\complete_radio_no_imageclassificationRF_opt-roc_auc_und-W\combination\measures_LR.pkl"    
    #file = r"\\amc.intra\users\L\laramos\home\Desktop\Combination_MrCLean\complete_radio_no_imageclassificationRF_opt-roc_auc_und-W\clinical_no_img_scores\measures_LR.pkl"
    #file = r"\\amc.intra\users\L\laramos\home\Desktop\Combination_MrCLean\complete_radio_no_imageclassificationRF_opt-roc_auc_und-W\clin_no_scores_with_autofeats\measures_LR.pkl"
    
    file = os.path.join(path,"measures_LR.pkl")
    #file = os.path.join(path,"measures_RFC.pkl")
    
    with open(file, 'rb') as f:
        lr_m = pickle.load(f)
    
    names = pd.read_csv(os.path.join(path,"columns_used.csv"))
    shap_values = lr_m.shap_values[0]
    for i in range(1,len(lr_m.shap_values)):
        shap_values  = np.concatenate((shap_values,lr_m.shap_values[i]),axis=0)
        
    # names = pd.read_csv(os.path.join(path,"columns_used.csv"))
    # shap_values = lr_m.shap_values[0][1]
    # for i in range(1,len(lr_m.shap_values)):
    #     shap_values  = np.concatenate((shap_values,lr_m.shap_values[i][1]),axis=0)        
        
    test_set = lr_m.test_sets[0]
    for i in range(1,len(lr_m.test_sets)):
        print(lr_m.test_sets[i].shape)
        test_set  = np.concatenate((test_set,lr_m.test_sets[i]), axis=0)
    
    better_names = list()
    for c in names['name']:
        c = c.replace('featoriginal_firstorder_','')
        better_names.append(c)
    
    test_set = pd.DataFrame(test_set, columns=better_names)

    shap_values = np.where(np.abs(shap_values)>5, 0, shap_values)

    shap.summary_plot(shap_values, test_set, plot_type="bar",show=True,auto_size_plot=(20,10))
    plt.tight_layout()
    #plt.savefig(os.path.join(folder,type_var+'bar.pdf'))
    #plt.savefig(os.path.join(folder,type_var+'bar.tiff'))


    shap.summary_plot(shap_values, test_set,show=True,auto_size_plot=(20,10))
    plt.tight_layout()
    #plt.savefig(os.path.join(folder,type_var+'dist.pdf'))
    #plt.savefig(os.path.join(folder,type_var+'dist.tiff'))

    

        
class Pipeline: 
 
    def RandomGridSearch(self,x_train,y_train,x_test,y_test,splits,path_results,m,itera,clf_g,name,tuned_parameters,opt,final_cols):
        """
        This function looks for the best set o parameters for RFC method
        Input: 
            X: training set
            Y: labels of training set
            splits: cross validation splits, used to make sure the parameters are stable
        Output:
            clf.best_params_: dictionary with the parameters, to use: param_svm['kernel']
        """    
        
        start_rfc = time.time()                  
        #clf_grid =  RandomizedSearchCV(clf_g, tuned_parameters, cv=splits,random_state=random_state,
        #                   scoring='%s' % opt[0],n_jobs=n_jobs)        
        clf_grid =  RandomizedSearchCV(clf_g, tuned_parameters, cv=splits,random_state=random_state,
                           scoring='%s' % opt[0],n_jobs=n_jobs,n_iter=50,verbose=1)
                                          
        clf_grid.fit(x_train, y_train)
        #print("Score",clf.best_score_)
        end_rfc = time.time()
        
        print("Time to process: ",end_rfc - start_rfc)
        
        with open(path_results+"//parameters_"+name+".txt", "a") as file:
            for item in clf_grid.best_params_:
              file.write(" %s %s " %(item,clf_grid.best_params_[item] ))
            file.write("\n")
            
        #clf = clf_g(**clf_grid.best_params_,random_state=random_state)
        clf = clf_grid.best_estimator_
        
        shape_visualization(x_test,m, clf,self.name)
        
                    
        #clf_t = clf_g(**clf_grid.best_params_,random_state=random_state)
        clf = clf.fit(x_train,y_train)

                             
        if name=="SVM":
            decisions = clf.decision_function(x_test)
            probas=\
            (decisions-decisions.min())/(decisions.max()-decisions.min())
        else:
             probas = clf.predict_proba(x_test)[:, 1]


                
        self.model = clf     
        
        preds = clf.predict(x_test) 
        precision, recall, _ = precision_recall_curve(y_test,preds)
        m.clf_f1_score[itera]=f1_score(y_test, preds)
        tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()        
        m.clf_sens[itera]=tp/(tp+fn)
        m.clf_spec[itera]=tn/(tn+fp)
        m.clf_ppv[itera]=tp/(tp+fp)
        m.clf_npv[itera]=tn/(tn+fn)                        
        m.clf_auc[itera] = roc_auc_score(y_test,probas)
        m.auc_prc[itera] = (auc(recall,precision))
        m.mcc[itera] = (matthews_corrcoef(y_test,preds))
        m.bacc[itera] = (balanced_accuracy_score(y_test,preds))
        fpr_rf, tpr_rf, _ = roc_curve(y_test, probas)  
        
        m.clf_brier[itera] = brier_score_loss(y_test, probas)   
                
        save_prob = np.concatenate((probas.reshape(-1,1),y_test.reshape(-1,1)),axis = 1)
        
        
      
        #np.save(path_results+"probabilities_"+name+"_"+str(itera)+".npy",probas)
        
        np.save(path_results+"//probabilities_"+name+"_"+str(itera)+".npy",save_prob)
        #np.save(path_results+"probabilities_train"+name+"_"+str(itera)+".npy",save_prob_train)

        #np.save(path_results+"feature_importance"+name+"_"+str(itera)+str(i)+".npy",clf.coef_)
        #joblib.dump(clf,path_results+'clf_'+name+str(itera)+str(i))
        return(fpr_rf,tpr_rf,probas,clf)
        
        
        
    def __init__(self,run,name ,x_train,y_train,x_test,y_test,itera,cv,mean_tprr,m,path_results,opt,und,final_cols):
        if run:
            opt=[opt]
            self.name = name
            if name == 'RFC':
                print("RFC Grid Search")                
                tuned_parameters = get_RFC()
                if und=='W':
                    clf = RandomForestClassifier(n_estimators=25, oob_score = True,random_state=random_state,class_weight='balanced')   
                else:
                    clf = RandomForestClassifier(n_estimators=25, oob_score = True,random_state=random_state)   
            else:
                if name == 'SVM':
                    print("SVM Grid Search")
                    tuned_parameters = get_SVC()
                    if und=='W':
                        clf = SVC(random_state=random_state,class_weight='balanced')
                    else:                    
                        clf = SVC(random_state=random_state)
                else:
                    if name == 'LR':
                        print("LR Grid Search")
                        tuned_parameters = get_LR()
                        if und=='W':
                            clf = LogisticRegression(random_state=random_state,max_iter=5000,class_weight='balanced') 
                        else:
                            clf = LogisticRegression(random_state=random_state,max_iter=5000) 
                    else:
                        if name == 'NN':
                            print("NN Grid Search")
                            tuned_parameters = get_NN()                                                        
                            clf = MLPClassifier(hidden_layer_sizes=(x_train.shape[1]),max_iter=5000,batch_size=32,random_state=random_state )                         
                                    
                        else:                    
                            if name == 'XGB':
                                print("XGB Grid Search")
                                tuned_parameters = get_XGB()
                                if und=='W':
                                    clf = xgb.XGBClassifier(random_state=random_state,scale_pos_weight=(y_train.shape[0]-sum(y_train))/sum(y_train)) 
                                else:                                    
                                    clf = xgb.XGBClassifier(random_state=random_state) 
                        
            self.name=name
            m.run=True
            fpr_rf,tpr_rf,probas_t,clf=self.RandomGridSearch(x_train,y_train,x_test,y_test,cv,path_results,m,itera,clf,name,tuned_parameters,opt,final_cols)
            print("Done Grid Search")
            print("Done testing - "+ name, m.clf_auc[itera])
            mean_fpr = np.linspace(0, 1, 100) 
            m.mean_tpr += interp(mean_fpr, fpr_rf,tpr_rf)
            m.mean_tpr[0] = 0.0
        else:
            self.name='NONE'
            self.clf=0



def Mean_Confidence_Interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.nanmean(a), sp.stats.sem(a,nan_policy = 'omit')
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m, m-h, m+h

def Print_Results_Excel(m,splits,names,path_results,l,data_to_use):    
    colors=['darkorange','blue','green','black','yellow']
    book = xlwt.Workbook(encoding="utf-8")    
    sheet1 = book.add_sheet("Sheet 1")
    #path_results_txt=path_results+path_results[2:len(path_results)-2]+str(l)+".xls"
    path_results_txt = os.path.join(path_results,('results'+str(l)+".xls"))
    sheet1.write(0, 0, "Methods")
    sheet1.write(0, 1, "AUC 95% CI ")
    sheet1.write(0, 2, "Brier ")
    sheet1.write(0, 3, "F1-Score")
    sheet1.write(0, 4, "Sensitivity")
    sheet1.write(0, 5, "Specificity")
    sheet1.write(0, 6, "PPV")
    sheet1.write(0, 7, "NPV")
    sheet1.write(0, 8, "AUPRC")
    sheet1.write(0, 9, "MCC")
    sheet1.write(0, 10, "BACC")
    #sheet1.write(0, 8, "F1-Score_f1")
    #sheet1.write(0, 9, "Sensitivity_f1")
    #sheet1.write(0, 10, "Specificity_f1")
    #sheet1.write(0, 11, "PPV_f1")
    #sheet1.write(0, 12, "NPV_f1")
    #sheet1.write(0, 13, "F1-Score_spec")
    #sheet1.write(0, 14, "Sensitivity_spec")
    #sheet1.write(0, 15, "Specificity_spec")
    #sheet1.write(0, 16, "PPV_spec")
    #sheet1.write(0, 17, "NPV_spec")
    #Spec and sensitivty are inverted because of the label
    for i in range(0,len(names)):        
        print(i,names[i])
        sheet1.write(i+1,0,(names[i])) 
        sheet1.write(i+1,1,str("%0.2f (%0.2f - %0.2f)"%(Mean_Confidence_Interval(m[i].clf_auc.reshape(-1)))))              
        sheet1.write(i+1,2,str("%0.2f (%0.2f - %0.2f)"%(Mean_Confidence_Interval(m[i].clf_brier.reshape(-1)))))              
        sheet1.write(i+1,3,str("%0.2f (%0.2f - %0.2f)"%(Mean_Confidence_Interval(m[i].clf_f1_score.reshape(-1)))))              
        sheet1.write(i+1,4,str("%0.2f (%0.2f - %0.2f)"%(Mean_Confidence_Interval(m[i].clf_sens.reshape(-1)))))              
        sheet1.write(i+1,5,str("%0.2f (%0.2f - %0.2f)"%(Mean_Confidence_Interval(m[i].clf_spec.reshape(-1)))))              
        sheet1.write(i+1,6,str("%0.2f (%0.2f - %0.2f)"%(Mean_Confidence_Interval(m[i].clf_ppv.reshape(-1)))))              
        sheet1.write(i+1,7,str("%0.2f (%0.2f - %0.2f)"%(Mean_Confidence_Interval(m[i].clf_npv.reshape(-1)))))              
        sheet1.write(i+1,8,str("%0.2f (%0.2f - %0.2f)"%(Mean_Confidence_Interval(m[i].auc_prc.reshape(-1)))))   
        sheet1.write(i+1,9,str("%0.2f (%0.2f - %0.2f)"%(Mean_Confidence_Interval(m[i].mcc.reshape(-1)))))   
        sheet1.write(i+1,10,str("%0.2f (%0.2f - %0.2f)"%(Mean_Confidence_Interval(m[i].bacc.reshape(-1)))))   
        
#        sheet1.write(i+1,8,str("%0.2f (%0.2f - %0.2f)"%(Mean_Confidence_Interval(m[i].f1_score_f1.reshape(-1)))))              
#        sheet1.write(i+1,9,str("%0.2f (%0.2f - %0.2f)"%(Mean_Confidence_Interval(m[i].sens_f1.reshape(-1)))))              
#        sheet1.write(i+1,10,str("%0.2f (%0.2f - %0.2f)"%(Mean_Confidence_Interval(m[i].spec_f1.reshape(-1)))))              
#        sheet1.write(i+1,11,str("%0.2f (%0.2f - %0.2f)"%(Mean_Confidence_Interval(m[i].clf_ppv_f1.reshape(-1)))))              
#        sheet1.write(i+1,12,str("%0.2f (%0.2f - %0.2f)"%(Mean_Confidence_Interval(m[i].clf_npv_f1.reshape(-1)))))              
#        
#        sheet1.write(i+1,13,str("%0.2f (%0.2f - %0.2f)"%(Mean_Confidence_Interval(m[i].f1_score_spec.reshape(-1)))))              
#        sheet1.write(i+1,14,str("%0.2f (%0.2f - %0.2f)"%(Mean_Confidence_Interval(m[i].sens_spec.reshape(-1)))))              
#        sheet1.write(i+1,15,str("%0.2f (%0.2f - %0.2f)"%(Mean_Confidence_Interval(m[i].spec_spec.reshape(-1)))))
#        sheet1.write(i+1,16,str("%0.2f (%0.2f - %0.2f)"%(Mean_Confidence_Interval(m[i].clf_ppv_spec.reshape(-1)))))              
#        sheet1.write(i+1,17,str("%0.2f (%0.2f - %0.2f)"%(Mean_Confidence_Interval(m[i].clf_npv_spec.reshape(-1)))))                                    

        np.save(file=os.path.join(path_results,('AUCs_'+names[i]+'.npy')),arr=m[i].clf_auc)
        #np.save(file=path_results+'Thresholds_'+names[i]+'.npy',arr=m[i].clf_thresholds)
        mean_tpr=m[i].mean_tpr
        mean_tpr /= splits
        mean_tpr[-1] = 1.0
        #frac_pos_rfc  /= skf.get_n_splits(X, Y)
        mean_fpr = np.linspace(0, 1, 100) 
        mean_auc_rfc = auc(mean_fpr, mean_tpr)
        plt.plot(mean_fpr, mean_tpr, color=colors[i],lw=2, label=names[i]+' (area = %0.2f)' % mean_auc_rfc)
        plt.legend(loc="lower right")
        np.save(file=os.path.join(path_results,('tpr_'+names[i]+'.npy')),arr=mean_tpr)
        np.save(file=os.path.join(path_results,('fpr_'+names[i]+'.npy')),arr=mean_fpr)
        if names[i]=='RFC':
            np.save(file=os.path.join(path_results,('Feat_Importance'+names[i]+'.npy')),arr=m[i].feat_imp)
        with open(os.path.join(path_results,'measures_'+names[i]+'.pkl'), 'wb') as f:
            pickle.dump(m[i],f)
    book.save(path_results_txt)        
    #plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')    
    #plt.show() 

