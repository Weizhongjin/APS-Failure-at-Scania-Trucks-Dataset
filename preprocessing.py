import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
data = pd.read_csv("../finaldata/aps_failure_training_set_SMALLER.csv" , na_values='na')

print(data.isna().sum(axis = 0))
#step 1 , process special feature, these feature have less than 20 numbers, i can treat them as category and set NaN as a class
data['ab_000']=data['ab_000'].fillna('missing')
data['as_000']=data['as_000'].fillna('missing')
data['cd_000']=data['cd_000'].fillna('missing')
data['ch_000']=data['ch_000'].fillna('missing')
data['ef_000']=data['ef_000'].fillna('missing')
#step 2, drop feature that miss 80% dat
data = data.dropna(thresh=19999*0.8,axis = 1) 

#step 3, fill missing data with mean of feature when these feature only have 2% missing data
for name in data.columns.values.tolist():
    if name == 'class':
        pass
    else:
        if data[name].isna().sum(axis = 0)<19999*0.02 and data[name].isna().sum(axis = 0) != 0:
            data[name] = data[name].fillna(round(data[name].mean()))
        pass

#step 4, set a new feature to save every sample's #missing data 
data['missing number'] = data.isna().sum(axis = 1)

#step 5, fill rest missing data by linear interpolate.
data = data.interpolate()

print(data.isna().sum(axis = 0))

for name in data.columns.values.tolist():
    if data[name].isna().sum(axis = 0)!= 0:
        print(name)

data.to_csv('../finaldata/preprocessed_data.csv')
