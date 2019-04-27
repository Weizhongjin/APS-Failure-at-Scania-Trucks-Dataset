import pandas as pd
import numpy as np
data = pd.read_csv("/Users/weizhongjin/usc/ee559/finaldata/aps_failure_training_set_SMALLER.csv" , na_values='na')
test_data = pd.read_csv("/Users/weizhongjin/usc/ee559/finaldata/aps_failure_test_set.csv" , na_values='na')
i=0
for name in data.columns.values.tolist():
    if data[name].isna().sum(axis = 0) > 19999*0.8:
        i+=1
print(i)