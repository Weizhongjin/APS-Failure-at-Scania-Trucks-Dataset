import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score,StratifiedKFold
from sklearn.metrics import accuracy_score,roc_curve,confusion_matrix,precision_recall_curve,auc,roc_auc_score,recall_score,classification_report
from sklearn.metrics import f1_score
import missingdata as md

missing_data_method = 'method1'
scaler_method = 'standard'
feature_choose = 'pca'
balance_method = 'SMOTE'
classifier_parameter = ['svm', [0.01,0.1] , [0.01], ['linear','rbf']]
foldN = 5
loop = 10

data = pd.read_csv("/Users/weizhongjin/usc/ee559/finaldata/aps_failure_training_set_SMALLER.csv" , na_values='na')
test_data = pd.read_csv("/Users/weizhongjin/usc/ee559/finaldata/aps_failure_test_set.csv" , na_values='na')

train_data, train_label , test_data, test_label = md.MissingData(data, test_data, missing_data_method)
train_data_scaler, test_data_scaler = md.Scaler(scaler_method,train_data,test_data)
train_data_selection, test_data_selection = md.Feature_selection(feature_choose, train_data_scaler, test_data_scaler )
train_data_final, train_label_final = md.Balance(balance_method,train_data_selection,train_label)
test_data_final = test_data_selection
final_classifier_parameter = md.Find_Best_Param(classifier_parameter, train_data_final, train_label_final,foldN)
md.Classifier(final_classifier_parameter,train_data_final,train_data_final,test_data_final,test_label,loop)