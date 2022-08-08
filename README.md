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
df = pd.read_csv(r"C:\Users\kttpa\Downloads\car_data - Sheet1.csv")
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
