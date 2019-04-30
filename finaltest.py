import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score,StratifiedKFold
from sklearn.metrics import accuracy_score,roc_curve,confusion_matrix,precision_recall_curve,auc,roc_auc_score,recall_score,classification_report
from sklearn.metrics import f1_score
import util as md

missing_data_method = ['method2']
preprocess_method = ['standard','pca']
balance_method = ['SMOTE',1]
classifier_parameter1 = ['KNN',2]
classifier_parameter2 = ['svm',1,0.01,'rbf']
classifier_parameter3 = ['NN',50]
foldN = 5
loop = 1
param = [0.95,60]
print('Start...')
data = pd.read_csv("/Users/weizhongjin/usc/ee559/finaldata/aps_failure_training_set.csv" , na_values='na')
test = pd.read_csv("/Users/weizhongjin/usc/ee559/finaldata/aps_failure_test_set.csv" , na_values='na')
print('-------------------------')
print('Parameter:')
scaler_method = preprocess_method[0]
feature_choose = preprocess_method[1]
train_data, train_label , test_data, test_label = md.MissingData(data, test, missing_data_method[0])
train_data = np.array(train_data)
test_data = np.array(test_data)
train_data, test_data = md.Scaler(scaler_method,train_data,test_data)
train_data, test_data = md.Feature_selection(feature_choose, train_data,train_label, test_data,param)
train_data, train_label = md.Balance(balance_method,train_data,train_label)  
print('/---------------------------------------/')
final_classifier_parameter = classifier_parameter3
md.Classifier(final_classifier_parameter, train_data, train_label,test_data,test_label,loop)

