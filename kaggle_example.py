import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score,StratifiedKFold
from sklearn.metrics import accuracy_score,roc_curve,confusion_matrix,precision_recall_curve,auc,roc_auc_score,recall_score,classification_report
from sklearn.metrics import f1_score
from sklearn import svm
data = pd.read_csv("/Users/weizhongjin/usc/ee559/finaldata/aps_failure_training_set_SMALLER.csv" , na_values='na')
test_data = pd.read_csv("/Users/weizhongjin/usc/ee559/finaldata/aps_failure_test_set.csv" , na_values='na')
train_label = data['class']
train_data = data.drop('class',axis=1)
test_label = test_data['class']
test_data = test_data.drop('class',axis=1)
before_drop = train_data.columns.values.tolist()
train_data = train_data.dropna(thresh=19999*0.3,axis = 1) 
after_drop = train_data.columns.values.tolist()
diff = set(before_drop).difference(after_drop)
for name in diff:
    test_data = test_data.drop(name,axis = 1)

train_data['missing number'] = train_data.isna().sum(axis = 1)
test_data['missing number'] = test_data.isna().sum(axis = 1)
for name in train_data.columns.values.tolist():
    if name == 'class':
        pass
    else:
        #if train_data[name].isna().sum(axis = 0)<19999*0.02 and train_data[name].isna().sum(axis = 0) != 0:
        if train_data[name].isna().sum(axis = 0) != 0:
            train_data[name] = train_data[name].fillna(train_data[name].median())
            test_data[name] = test_data[name].fillna(train_data[name].median())
        pass

train_label = train_label.apply(lambda x: 0 if x=='neg' else 1)
test_label = test_label.apply(lambda x: 0 if x=='neg' else 1)

scaler = StandardScaler()
#scaler = Mydeep().Choose_Scaler('StandardScaler')
scaler.fit(train_data)
train_data_scaler = scaler.transform(train_data)
test_data_scaler = scaler.transform(test_data)
pca = PCA(0.95)
pca.fit(train_data_scaler)
train_data_pca = pca.transform(train_data_scaler)
train_data_preprocessed = pd.DataFrame(train_data_pca)
test_data_pca = pca.transform(test_data_scaler)
test_data_preprocessed = pd.DataFrame(test_data_pca)

sm = SMOTE(ratio = "minority")
train_data_balance, train_label_balance = sm.fit_sample(train_data_pca, train_label)
train_data_final = pd.DataFrame(train_data_balance)
train_label_final = pd.Series(train_label_balance)
print(train_label_final.value_counts())

cv = StratifiedKFold(n_splits=5)
score_vec = np.zeros(20)
c_vec = np.zeros(20)
ken_vec = np.zeros(20)
i = 0
#---------------Logistic Regression-------------------
# c_parameter_range = [0.0001,0.001,0.01,0.1,1,10,100]
# penalty = ['l1','l2']
c_parameter_range = [0.001,0.01,0.1,10,100]
kernels = ['linear','poly','rbf','sigmoid']
#---------------SVM-----------------------------------
for kernel in kernels:
    if kernel == 'linear':
        kernel_n = 1
    elif kernel == 'poly':
        kernel_n = 2
    elif kernel == 'rbf':
        kernel_n = 3
    elif kernel == 'sigmod':
        kernel_n = 4
        
    for c_param in c_parameter_range:
        print('/---------------------------------------------/')
        auc_score = 0
        print('------------------------')
        print("C Parameter :", c_param)
        print("kernel: ", kernel)
        print('------------------------')
        for train, val in cv.split(train_data_balance, train_label_balance):

            clf = svm.SVC(C = c_param,kernel = kernel,gamma = 0.01)
            clf.fit(train_data_balance[train],train_label_balance[train])
            y_pred = clf.predict(train_data_balance[val])
            Recall = roc_auc_score(train_label_balance[val],y_pred)
            auc_score += Recall
            
        auc_score = auc_score/5
        
        score_vec[i] = auc_score
        c_vec[i] = c_param
        ken_vec[i] = kernel_n
        i += 1
        print ('Recall score for c param', c_param,'and kerne;',kernel,'=',auc_score)
        print('-------------------------')
        print('')

# for penal in penalty:
#     if penal == 'l1':
#         penal_n = 1
#     else:
#         penal_n = 2
#     for c_param in c_parameter_range:
#         print('/---------------------------------------------/')
#         auc_score = 0
#         print('------------------------')
#         print("C Parameter :", c_param)
#         print("Penalty: ", penal)
#         print('------------------------')
#         for train, val in cv.split(train_data_balance, train_label_balance):

#             lr = LogisticRegression(C = c_param, penalty = penal,solver='liblinear')
#             lr.fit(train_data_balance[train],train_label_balance[train])
#             y_pred = lr.predict(train_data_balance[val])
#             Recall = roc_auc_score(train_label_balance[val],y_pred)
#             auc_score += Recall
            
#         auc_score = auc_score/5
        
#         score_vec[i] = auc_score
#         c_vec[i] = c_param
#         pen_vec[i] = penal_n
#         i += 1
#         print ('Recall score for c param', c_param,'and penalty',penal,'=',auc_score)
#         print('-------------------------')
#         print('')


# ind_max = np.where(score_vec == np.max(score_vec))
# best_c = int(c_vec[ind_max])
# best_penalty = pen_vec[ind_max]
# if best_penalty == 1:
#     best_penalty = 'l1'
# else:
#     best_penalty = 'l2'
# lr_best = LogisticRegression(C = best_c, penalty = best_penalty,solver='liblinear')
# lr_fit=lr_best.fit(train_data_balance, train_label_balance)
# y_pred_test = lr_best.predict(test_data_pca)
# recall_test_lr=roc_auc_score(test_label,y_pred_test)
# print ('Final Recall score for c param', c_param,'and penalty',penal,'=',recall_test_lr)
# print('-------------------------')
# print('')
# cm = confusion_matrix(test_label,y_pred_test).ravel()
# cm = pd.DataFrame(cm.reshape((1,4)), columns=['TN', 'FP', 'FN', 'TP'])
# print(cm.info())
# print(cm.head())
# print(cm)
# false_positive_rate, true_positive_rate, thresholds = roc_curve(test_label,y_pred_test)
# roc_auc = auc(false_positive_rate, true_positive_rate)
ind_max = np.where(score_vec == np.max(score_vec))
best_c = int(c_vec[ind_max])
best_kernel = ken_vec[ind_max]
if best_kernel == 1:
    best_kernel = 'linear'
elif best_kernel == 2:
    best_kernel = 'ploy'
elif best_kernel == 3:
    best_kernel = 'rbf'
else:
    best_kernel = 'sigmod'  
svm_best = svm.SVC(C =best_c,gamma = best_kernel, kernel = 'sigmoid')
svm_fit=svm_best.fit(train_data_balance, train_label_balance)
y_pred_test = svm_best.predict(test_data_pca)
recall_test_svm=roc_auc_score(test_label,y_pred_test)
print ('Final Recall score for c param', best_c,'and kernel',best_kernel,'=',recall_test_svm)
print('-------------------------')
print('')
cm = confusion_matrix(test_label,y_pred_test).ravel()
cm = pd.DataFrame(cm.reshape((1,4)), columns=['TN', 'FP', 'FN', 'TP'])
print(cm.info())
print(cm.head())
print(cm)
false_positive_rate, true_positive_rate, thresholds = roc_curve(test_label,y_pred_test)
roc_auc = auc(false_positive_rate, true_positive_rate)

plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate, true_positive_rate, 'b',
label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
cost = 500*cm['FN']+10*cm['FP']
print('The final cost is : {}'.format(cost))