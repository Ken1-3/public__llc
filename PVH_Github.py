# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 14:20:27 2022

@author: kttpa
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 14:05:03 2022

@author: kttpa
"""
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn import svm

#_______________________________________________________________________
# NO.1 
#_______________________________________________________________________

#Read .CSV as a table
df = pd.read_csv(r"C:\Users\kttpa\Downloads\car_data - Sheet1.csv")

#list of column names obtained from data instructions
column_names_list = ['mpg','cylinders','displacement','horsepower','weight','acceleration','model_year','origin','car_name']
df.columns = column_names_list

#drop qualitiative column
df = df.drop(['car_name'], axis=1)

#clean the dataset from problematic values
issue_list = []
location_list = []

#list to scan for values
remove_list = [',',':','/',' ','>',';','?']

#filter out any bad data
for clean in df.itertuples():   
    for i in clean:
        if i in remove_list:
            issue_list.append(i)
            location_list.append(clean.Index)
  
#Create a zipped list with the rows with data integrity issues            
final = zip(issue_list,location_list)

#print the issue rows 
for j,k in final:
    print('Issue Character Found:',j,'  |  Deleting Row:',k)
    
    """Issue Character Found: ?   |  Deleting Row: 31
       Issue Character Found: ?   |  Deleting Row: 125
       Issue Character Found: ?   |  Deleting Row: 329
       Issue Character Found: ?   |  Deleting Row: 335
       Issue Character Found: ?   |  Deleting Row: 353
       Issue Character Found: ?   |  Deleting Row: 373"""

#Delete the issue rows
df = df.drop(index = location_list)

#copy df as another variable
sep_df = df
    

#_______________________________________________________________________
# NO.2
#_______________________________________________________________________


X = df[['cylinders','displacement','horsepower','weight','acceleration','origin']].values
y = df['mpg'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# Step 3: Using your training data run a linear regression using mpg as your predicted variable
# and cylinders, displacement, horsepower, weight, and acceleration as your predictor variables.


#_______________________________________________________________________
# NO.3
#_______________________________________________________________________

regressor = LinearRegression()  
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

df_new = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

df1 = df_new.head(5)

print(df1)

df1.plot(kind='bar',figsize=(10,8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()


# Step 4: Using your testing data, calculate the mean squared error (MSE) of your model. As a
# bonus perform 10 fold cross validation if you can.


print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred)) 

from sklearn.model_selection import cross_val_score

from sklearn import preprocessing
from sklearn import utils

lab = preprocessing.LabelEncoder()
y_transformed = lab.fit_transform(y)

MAX_CV_FOLDS = 10

# Ensure we have data we can use for MAX_CV_FOLDS
df["CLASS_COUNT"] = df["mpg"]
df["CLASS_COUNT"] = df.groupby("mpg").transform("count")
good_df = df[df["CLASS_COUNT"] >= MAX_CV_FOLDS]
good_X = good_df[['cylinders','displacement','horsepower','weight','acceleration']].values
good_y = good_df['mpg'].values
good_y_transformed = lab.fit_transform(good_y)

clf = svm.SVC(kernel='linear', C=1,)
scores = cross_val_score(clf,good_X,good_y_transformed, cv=MAX_CV_FOLDS)

print('-----------------------------------------')

print(scores)

print('-----------------------------------------')

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

#https://www.kdnuggets.com/2019/03/beginners-guide-linear-regression-python-scikit-learn.html/2




# Step 5 (bonus): Transform mpg into a binary variable (splitting at the median to define “high”
# mpg and “low” mpg as 1, 0). Perform steps 1-4 again but instead of MSE use prediction
# accuracy as your performance metric and logistic regression as your model choice.


import statistics
mpg_list = sep_df['mpg'].to_list()


res = statistics.median(mpg_list)

for i in sep_df.itertuples():
    if i.mpg < res:
        sep_df.at[(i.Index),'mpg'] = 0
        
    else:
        sep_df.at[(i.Index),'mpg'] = 1
        

print(sep_df.head(5))
        
X = sep_df[['cylinders','displacement','horsepower','weight','acceleration']].values
y = sep_df['mpg'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)

from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))


