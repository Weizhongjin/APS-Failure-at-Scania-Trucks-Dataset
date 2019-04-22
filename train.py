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

class Mydeep:
    def __init__(self):
        self.scaler_method = 'StandardScaler'
        self.dimension_reduce = 'PCA'
        self.scaler = StandardScaler()
        self.reducer = PCA()
    def Choose_Scaler(self, scaler_method):
        self.scaler_method = 'StandardScaler'
        if self.scaler_method == 'StandardScaler':
            self.scaler = StandardScaler()
        else:
            self.scaler = MinMaxScaler()
        return self.scaler
    def Choose_D_Reduce(self, reduce_method):
        self.dimension_reduce = reduce_method
        if self.dimension_reduce == 'PCA':
            self.reducer = PCA()
            pass
        else self.dimension_reduce == 'SelectKbest':
            self.reducer = SelectKBest()
            pass