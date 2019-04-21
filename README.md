## EE559 Final-Project APS Failure at Scania Trucks Dataset

### Project Information:

​	The dataset consists of data collected from heavy Scania trucks in everyday usage.The system in focus on the “Air Pressure system (APS) which generates pressurized air that are utilized in various functions in a truck, such as braking and gear changes.” [1]. This is a 2-class problem, and the goal is to predict the failure of components in the APS system, given various inputs. 



### Data:

Training and testing are posted on [D2L](https://courses.uscden.net/d2l/le/content/15346/Home?itemIdentifier=D2L.LE.Content.ContentObject.ModuleCO-251275); the training set labeled “SMALLER” has been down-sampled by factor of 3 (stratified), from the complete training set. The complete training set is also posted on D2L in case you choose to work with it also.* Please use the complete test set, as posted on D2L, for your test set. (There is no down-sampled version of the test set.)
For more information on the dataset:
https://archive.ics.uci.edu/ml/datasets/APS+Failure+at+Scania+Trucks



### Data Preprocessing

#### Data missing:

1. When a feature have a lot of missing data, we can directly drop this feature.

   Code:

   we can see how much data is missing within a feature at first:

   `print(data.isna().sum(axis = 0))`

   then drop features that missing 70% data

   `print(data.dropna(thresh=19999*0.7,axis = 1))`

2. When the missing rate is under 10%, we can use methods below:

   1. We can drop sample that misses some feature.(this method may not try)

      Code:

      `data.dropna()`

      when one sample miss half of its feature data

   2. Treat missing as a class of feature.

      Code:

      `data.fillna('missing')`

   3. Using the mean/mode/median of existed datas to fill these missing datas.

      Code:

      `data.fillna(data.mean())`, `dff.fillna(dff.mean()['B':'C'])`(focus on some columns)

   4. Using KNN or fill with next(bfill) or last(pad) data.

      Code:

      `data.fillna(#)`

      has `method='pad'` , `method='bfill' `

   5. Using interpolate method.

      Code:

      `data.interpolate()`

      how to use interpolate please check at [pandas document](<http://pandas.pydata.org/pandas-docs/stable/user_guide/missing_data.html>).

3. Train a model to predict this miss data.

   We can use R programming to solve this problem with mice package.（difficult, i will try this method after I finished the baseline)
   
4. Finally, I will try the finally method as :

   a.	when missing data rate > 60%, Drop this feature.

   b.	when missing data rate < 5%,  i will use mean to fill them.

   c.	others rate , I will use interpolate to fill them. What's more, if that feature have less than 20 classes, we can sat nah as a new class. And use onehotencoding to encode these features.

   d.	I will a new feature that will show how many data miss within this sample after I finish drop and fill by mean.

   ```data['missing number'] = data.isna().sum(axis = 1)```

   e.	For rest missing data, I will interpolate method to fill them.

### Problem in this Project

1. pandas can't recognize the missing data "na", the regular missing data will present as "NaN" or "None" etc.

   I used 

   `data = pd.read_csv("../finaldata/aps_failure_training_set_SMALLER.csv"')`

   at first, but these code will not treat na in csv file as NaN(NULL)

   Solution:

   `data = pd.read_csv("../finaldata/aps_failure_training_set_SMALLER.csv" , na_values='na')`
   
2. To check which feature has less than 20 classes, i use R studio to explore by code:

   ```
   data<-read.csv('aps_failure_training_set_SMALLER.csv',na.strings = 'na')
   for (i in 2:ncol(data)) {
     if(length(table(data[i])) < 20 )
     {
       print(table(data[i]))
       print(names(data[i]))
     }
   }
   ```

   we can use one hot encoder on these features.

   These features and their classes are :

   ```
   [1] "ab_000"
      0    2    4    6    8   10   12   14   16   18   20   22   26   34   52   68 
   3626  654  128   42   22   14   10    3    1    2    1    2    1    2    1    1 
   
   [1] "as_000"
         0       4    6268   21748  128844  489122  693990 1246190 1655240 
     19760       1       1       1       1       1       1       1       1 
   
   [1] "cd_000"
   1209600 
     19753 
   
   [1] "ch_000"
       0     2 
   15019     5 
   
   [1] "ef_000"
       0     2     4     6     8    10    12    14    26    74   144   166   276   320 
   18972    47    21     7     2     2     2     2     1     1     2     1     1     1 
   ```

   