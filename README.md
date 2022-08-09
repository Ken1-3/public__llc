# Takehome

## Question 1 

#### Importing the required modules for the baseline questions:
```import pandas as pd 
import numpy as np  
import matplotlib.pyplot as plt 
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split  
from sklearn.linear_model import LinearRegression 
from sklearn import metrics 
```
#### We must first import the dataset into the file
```
#Read .CSV as a table
df = pd.read_csv(r"file_path")
```
#### Since column names were not assigned in raw data, we will assign them here - discovered from associative files
```
column_names_list = ['mpg','cylinders','displacement','horsepower','weight','acceleration','model_year','origin','car_name']
df.columns = column_names_list
```

#### Based on instructions, we remove qualitative properties
```
df = df.drop(['car_name'], axis=1)
```

#### We will also filter out any bad data that may cause issues with the dataset

```
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
```

## Question 2

#### Splitting into X,Y values
```
X = df[['cylinders','displacement','horsepower','weight','acceleration']].values
y = df['mpg'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
```

## Question 3

#### Getting Results

```
regressor = LinearRegression()  
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

![image](https://user-images.githubusercontent.com/89386946/183564431-720ae0c8-cfb1-4fc3-b20c-a09c8ece0730.png)

df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

df1.plot(kind='bar',figsize=(10,8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()

![image](https://user-images.githubusercontent.com/89386946/183564296-f096206c-11ef-4c40-9794-bb68388cb241.png)

```

## Question 4
#### MSE & Cross Validation 

```
mse = metrics.mean_squared_error(y_test, y_pred)

![image](https://user-images.githubusercontent.com/89386946/183564197-ae0127c3-58ec-4c51-b90d-e0e5b924c37d.png)


from sklearn.model_selection import cross_val_score

clf = svm.SVC(kernel='linear', C=1, random_state=42)
scores = cross_val_score(clf, X, y, cv=5)
![image](https://user-images.githubusercontent.com/89386946/183563920-b6f85fac-4012-4f72-bf25-d3010ff0d7fc.png)
```

## Question 5
#### Binary CLassification Model

```
import statistics

mpg_list = sep_df['mpg'].to_list()

res = statistics.median(mpg_list)

for i in sep_df.itertuples():
    if i.mpg < res:
        print(i.mpg)
        sep_df.at[(i.Index),'mpg'] = 0
        
    else:
        sep_df.at[(i.Index),'mpg'] = 1
 
 ![image](https://user-images.githubusercontent.com/89386946/183563801-e6e9655c-2e75-4ce6-aeed-01cd1ed26171.png)
```

 From there we run this dataframe through the series scripted above.
 
