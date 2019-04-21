import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
data = pd.read_csv("../finaldata/aps_failure_training_set_SMALLER.csv" , na_values='na')
print(data.fillna(data.mean()))
#print(data.dropna(thresh=19999*0.7,axis = 1))


