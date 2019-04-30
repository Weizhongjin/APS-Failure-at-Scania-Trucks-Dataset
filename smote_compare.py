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

missing_data_methods = ['method2']
#preprocess_methods = [['standard','pca'],['minmax','KBest']]
#missing_data_method = ['method2']
preprocess_method = ['standard','pca']
balance_method = ['SMOTE']
# balance_method = ['No',0]

# classifier_parameter = ['svm',[1],[0.01],['rbf']] #classifier 1
#classifier_parameter = ['SGDperceptron',['l1','l2']]
#classifier_parameter =['logisticRegression',[0.1,1,10,100],['l2']]
classifier_parameter1 = ['KNN',[2]]
classifier_parameter2 = ['NN',[50]]
classifier_parameter3 = ['svm',[1],[0.01],['rbf']] 
#classifier_parameter =['NB',['gaussian','ber']]
foldN = 5
loop = 1
cost_vector1 = []
cost_rate_vector1 = []
score_vector1 = []
cost_vector2 = []
cost_rate_vector2 = []
score_vector2 = []
cost_vector3 = []
cost_rate_vector3 = []
score_vector3 = []
param1 = [0.95,60]
params =range(19,99,2)
i = 1
print('Start...')
data = pd.read_csv("/Users/weizhongjin/usc/ee559/finaldata/aps_failure_training_set_SMALLER.csv" , na_values='na')
test = pd.read_csv("/Users/weizhongjin/usc/ee559/finaldata/aps_failure_test_set.csv" , na_values='na')
for missing_data_method in missing_data_methods:
    for param in params:
    #for preprocess_method in preprocess_methods:
        param = param/100
        balance_method_all = [balance_method,param]
        print('-------------------------')
        print('Parameter:')
        scaler_method = preprocess_method[0]
        feature_choose = preprocess_method[1]
        train_data, train_label , test_data, test_label = md.MissingData(data, test, missing_data_method)
        train_data = np.array(train_data)
        test_data = np.array(test_data)
        train_data, test_data = md.Scaler(scaler_method,train_data,test_data)
        train_data, test_data = md.Feature_selection(feature_choose, train_data,train_label, test_data,param1)
        train_data, train_label = md.Balance(balance_method_all,train_data,train_label)  
        final_classifier_parameter1,cost_vec1,cost_rate_vec1,score_vec1 = md.Find_Best_Param(classifier_parameter1, train_data, train_label,foldN)
        final_classifier_parameter2,cost_vec2,cost_rate_vec2,score_vec2 = md.Find_Best_Param(classifier_parameter2, train_data, train_label,foldN)
        final_classifier_parameter3,cost_vec3,cost_rate_vec3,score_vec3 = md.Find_Best_Param(classifier_parameter3, train_data, train_label,foldN)
        cost_vector1.append(cost_vec1)
        cost_rate_vector1.append(cost_rate_vec1[0])
        score_vector1.append(score_vec1)
        cost_vector2.append(cost_vec2)
        cost_rate_vector2.append(cost_rate_vec2[0])
        score_vector2.append(score_vec2)
        cost_vector3.append(cost_vec3)
        cost_rate_vector3.append(cost_rate_vec3[0])
        score_vector3.append(score_vec3)
        # print('Best classifier parameter :{}'.format(final_classifier_parameter))
        # print('Cost vector: {}'.format(cost_vec))
        # print('Cost Rate vector: {}'.format(cost_rate_vec))
        # print('Score vector: {}'.format(score_vec))
print('/---------------------------------------/')
cost_rate_vectors1 = np.array(cost_vector1)  
cost_rate_vectors2 = np.array(cost_vector2)  
cost_rate_vectors3 = np.array(cost_vector3)  
plt.title('Comparison of different Smot Parameter')
plt.plot(params,cost_rate_vectors1,color='blue', label='KNN')
plt.plot(params,cost_rate_vectors2,color='red', label='SVM')
plt.plot(params,cost_rate_vectors3,color='green', label='MLP(NN)')
plt.legend() 
plt.xlabel('Smote Ratio')
plt.ylabel('Cost')
plt.show()
