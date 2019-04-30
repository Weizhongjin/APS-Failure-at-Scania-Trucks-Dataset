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

missing_data_methods = ['method1','method2']
preprocess_methods = [['standard','pca'],['minmax','KBest']]
#missing_data_method = ['method2']
#preprocess_method = ['standard','pca']
balance_method = ['SMOTE',1]
# balance_method = ['No',0]

# classifier_parameter = ['svm',[0.1,1,10,100],[0.01],['rbf','linear','sigmoid']] #classifier 1
#classifier_parameter = ['SGDperceptron',['l1','l2']]
classifier_parameter =['logisticRegression',[0.1,1,10,100],['l2']]
#classifier_parameter =['NB',['gaussian','ber']]
#classifier_parameter =['NN',[50,100,150]]
foldN = 5
loop = 1
param1 = [0.95,60]
params =range(19,99,2)
i = 1
print('Start...')
data = pd.read_csv("/Users/weizhongjin/usc/ee559/finaldata/aps_failure_training_set_SMALLER.csv" , na_values='na')
test = pd.read_csv("/Users/weizhongjin/usc/ee559/finaldata/aps_failure_test_set.csv" , na_values='na')
for missing_data_method in missing_data_methods:
    for preprocess_method in preprocess_methods:
        print('-------------------------')
        print('Parameter:')
        scaler_method = preprocess_method[0]
        feature_choose = preprocess_method[1]
        train_data, train_label , test_data, test_label = md.MissingData(data, test, missing_data_method)
        train_data = np.array(train_data)
        test_data = np.array(test_data)
        train_data, test_data = md.Scaler(scaler_method,train_data,test_data)
        train_data, test_data = md.Feature_selection(feature_choose, train_data,train_label, test_data,param1)
        train_data, train_label = md.Balance(balance_method,train_data,train_label)  
        final_classifier_parameter,cost_vec,cost_rate_vec,score_vec = md.Find_Best_Param(classifier_parameter, train_data, train_label,foldN)
        print('Best classifier parameter :{}'.format(final_classifier_parameter))
        print('Cost vector: {}'.format(cost_vec))
        print('Cost Rate vector: {}'.format(cost_rate_vec))
        print('Score vector: {}'.format(score_vec))
print('/---------------------------------------/')

