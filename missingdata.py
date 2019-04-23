import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression, perceptron, SGDClassifier
from sklearn.model_selection import cross_val_score,StratifiedKFold
from sklearn.metrics import accuracy_score,roc_curve,confusion_matrix,precision_recall_curve,auc,roc_auc_score,recall_score,classification_report
from sklearn.metrics import f1_score
from sklearn import svm
from sklearn.naive_bayes import GaussianNB

def MissingData(data, test_data, method):
    if method == 'method1':
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
    
    elif method == 'method2':
        train_label = data['class']
        train_data = data.drop('class',axis=1)
        test_label = test_data['class']
        test_data = test_data.drop('class',axis=1)
        before_drop = train_data.columns.values.tolist()
        train_data = train_data.dropna(thresh=19999*0.2,axis = 1) 
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
                    train_data[name] = train_data[name].fillna(train_data[name].mean())
                    test_data[name] = test_data[name].fillna(train_data[name].mean())
                pass

        train_label = train_label.apply(lambda x: 0 if x=='neg' else 1)
        test_label = test_label.apply(lambda x: 0 if x=='neg' else 1)
    return train_data, train_label , test_data, test_label

def Scaler(scaler_method,train_data,test_data):
    if scaler_method == 'standard':
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()
    scaler.fit(train_data)
    train_data_scaler = scaler.transform(train_data)
    test_data_scaler = scaler.transform(test_data)
    return train_data_scaler, test_data_scaler
    
def Feature_selection(feature_choose, train_data_scaler,test_data_scaler):
    if feature_choose == 'pca':
        selector = PCA(0.95)

        pass
    else:
        pass

    selector.fit(train_data_scaler)
    train_data_selection = selector.transform(train_data_scaler)
    test_data_selection = selector.transform(test_data_scaler)
    return train_data_selection, test_data_selection

def Balance(balance_method, train_data, train_label):
    if balance_method == 'SMOTE':
        balancer = SMOTE(ratio = "minority")
        pass
    else:
        pass
    train_data_final, train_label_final = balancer.fit_sample(train_data, train_label)
    return train_data_final, train_label_final

def Find_Best_Param(classifier_parameter, train_data_final, train_label_final,foldN):
    clf_kind = classifier_parameter[0]
    parameter = classifier_parameter[1:]
    cv = StratifiedKFold(n_splits=foldN)
    score_vec = []
    if clf_kind == 'svm':
        c_param = parameter[0]
        gamma_param = parameter[1]
        kernel_param = parameter[2]
        c_vec = []
        gamma_vec = []
        kernel_vec = []
        output_parameter = [clf_kind]
        for c in c_param:
            for gam in gamma_param:
                for kern in kernel_param:
                    print('/---------------------------------------------/')
                    auc_score = 0
                    print('------------------------')
                    print("C Parameter :", c)
                    print("Gamma: ", gam)
                    print("kernel: ", kern)
                    print('------------------------')
                    for train, val in cv.split(train_data_final, train_label_final):
                        clf = svm.SVC(C=c,gamma=gam,kernel=kern)
                        clf.fit(train_data_final[train],train_label_final[train])
                        y_pred = clf.predict(train_data_final[val])
                        Recall = roc_auc_score(train_label_final[val],y_pred)
                        auc_score += Recall
                    auc_score = auc_score/foldN
                    score_vec.append(auc_score)
                    c_vec.append(c)
                    kernel_vec.append(kern)
                    gamma_vec.append(gam)
                    print ('Recall score for c param', c_param,'and kerne',kern,'and gamma',gam , '  =',auc_score)
                    print('-------------------------')
                    print('')
        ind_max = score_vec.index(max(score_vec))
        best_c = c_vec[ind_max]
        best_gamma = gamma_vec[ind_max]
        best_kernel = kernel_vec[ind_max]
        output_parameter.append(best_c)
        output_parameter.append(best_gamma)
        output_parameter.append(best_kernel)
    elif clf_kind == 'logisticRegression':
        c_param = parameter[0]
        penaltys = parameter[1]
        c_vec = []
        penalty_vec = []
        output_parameter = [clf_kind]
        for c in c_param:
            for penal in penaltys:
                print('/---------------------------------------------/')
                auc_score = 0
                print('------------------------')
                print("C Parameter :", c)
                print("Penalty: ", penal)
                print('------------------------')
                for train, val in cv.split(train_data_final, train_label_final):
                    clf = LogisticRegression(C = c, penalty = penal ,solver='liblinear')
                    clf.fit(train_data_final[train],train_label_final[train])
                    y_pred = clf.predict(train_data_final[val])
                    Recall = roc_auc_score(train_label_final[val],y_pred)
                    auc_score += Recall
                auc_score = auc_score/foldN
                score_vec.append(auc_score)
                c_vec.append(c)
                penalty_vec.append(penal)
                print ('Recall score for c param', c_param,'and Penalty',penal,'=',auc_score)
                print('-------------------------')
                print('')    
        ind_max = score_vec.index(max(score_vec))
        best_c = c_vec[ind_max]
        best_penalty = penalty_vec[ind_max]
        output_parameter.append(best_c)
        output_parameter.append(best_penalty)
    elif clf_kind == 'SGDperceptron':
        penaltys = parameter[1]
        penalty_vec = []
        for penalty_n in penaltys:
            print('/---------------------------------------------/')
            auc_score = 0
            print('------------------------')
            print("Penalty: ", penalty_n)
            print('------------------------')
            for train, val in cv.split(train_data_final, train_label_final):
                clf = SGDClassifier(loss='perceptron', penalty= penalty_n)
                clf.fit(train_data_final[train],train_label_final[train])
                y_pred = clf.predict(train_data_final[val])
                Recall = roc_auc_score(train_label_final[val],y_pred)
                auc_score += Recall
            auc_score = auc_score/foldN
            score_vec.append(auc_score)
            penalty_vec.append(penalty_n)
            print ('Recall score for Penalty',penalty_n,'=',auc_score)
            print('-------------------------')
            print('')    
        ind_max = score_vec.index(max(score_vec))
        best_penalty = penalty_vec[ind_max]
        output_parameter.append(best_penalty)
    elif clf_kind == 'GaussianNB':
        clf = GaussianNB()
    else:
        clf = svm.SVC(C=0.01,gamma=0.01,kernel='rbf')
        output_parameter.append(0.01)
        output_parameter.append(0.01)
        output_parameter.append('rbf')
    return output_parameter
def Classifier(classifier_parameter, train_data_final, train_label_final, text_data_final, test_label, loop):
    clf_kind = classifier_parameter[0]
    parameter = classifier_parameter[1:-1]
    final = 0
    cm_final = np.zeros(4)
    for i in range(loop):
        if clf_kind == 'svm':
            clf = svm.SVC(C=parameter[0],gamma=parameter[1],kernel=parameter[2])
        elif clf_kind == 'logisticRegression':
            clf = LogisticRegression(C = parameter[0], penalty = parameter[1] ,solver='liblinear')
        elif clf_kind == 'SGDperceptron':
            clf = SGDClassifier(loss='perceptron', penalty= parameter[1])
        elif clf_kind == 'GaussianNB':
            clf = GaussianNB()
        else:
            clf = svm.SVC(C=0.01,gamma=0.01,kernel='rbf')

        clf_fit = clf.fit(train_data_final, train_label_final)
        y_pred_test = clf.predict(text_data_final)
        recall_test_per=roc_auc_score(test_label,y_pred_test)
        final += recall_test_per
        cm = confusion_matrix(test_label,y_pred_test).ravel()
        cm_final += cm
    final_score = final/loop
    cm_final = round(cm_final/loop)
    print ('Final Recall score for Classifier=',final_score)
    print('-------------------------')
    print('')
    cm_final = pd.DataFrame(cm_final.reshape((1,4)), columns=['TN', 'FP', 'FN', 'TP'])
    print(cm_final.info())
    print(cm_final.head())
    print(cm_final)
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


