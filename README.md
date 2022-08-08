# Takehome

### Importing the required modules for the baseline questions:
```import pandas as pd 
import numpy as np  
import matplotlib.pyplot as plt 
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split  
from sklearn.linear_model import LinearRegression 
from sklearn import metrics 
```
### We must first import the dataset into the file
```
#Read .CSV as a table
df = pd.read_csv(r"C:\Users\kttpa\Downloads\car_data - Sheet1.csv")
```
### Since column names were not assigned in raw data, we will assign them here - discovered from associative files
```
column_names_list = ['mpg','cylinders','displacement','horsepower','weight','acceleration','model_year','origin','car_name']
df.columns = column_names_list
```
