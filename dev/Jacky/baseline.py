#Import libraries
import csv
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# Set the randomizer seed so results are the same each time.
np.random.seed(0)

#Load data
crime = pd.read_csv('../Data/train.csv')

#Recode weekday variable
weekday_mapping =  {'Monday': 1,
                    'Tuesday': 2,
                    'Wednesday': 3,
                    'Thursday': 4,
                    'Friday': 5,
                    'Saturday': 6,
                    'Sunday': 7}
crime['DayOfWeek']=crime['DayOfWeek'].map(weekday_mapping)

#Recode PdDistrict variable
enc = preprocessing.LabelEncoder()
district = enc.fit_transform(crime.PdDistrict)
crime['District'] = district

#Scale the coordinates variables
#stdsc = StandardScaler()
#crime['X_stdsc'] = stdsc.fit_transform(crime['X'])
#crime['Y_stdsc'] = stdsc.fit_transform(crime['Y'])


#Extract date and time information from 'Dates' variable
crime['Dates'] = pd.to_datetime(crime['Dates'])
crime['Year'] = crime['Dates'].dt.year
crime['Month'] = crime['Dates'].dt.month
crime['Day'] = crime['Dates'].dt.day
crime['Hour'] = crime['Dates'].dt.hour
crime['Minute'] = crime['Dates'].dt.minute


#Shuffle data
crime = crime.reindex(np.random.permutation(crime.index))


X = crime[['Year', 'Month', 'Day', 'Hour', 'Minute', 'DayOfWeek', 'District', 'X','Y']]
Y = crime.Category

test_data, test_labels = X[710000:], Y[710000:]
dev_data, dev_labels = X[700000:710000], Y[700000:710000]
train_data, train_labels = X[:700000], Y[:700000]

def RF():
    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(train_data, train_labels)
    print ('Accuracy: %3.3f' %(rf.score(dev_data,dev_labels)))

RF()
