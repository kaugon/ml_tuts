######################################
## Copyright 2017 @ KauGon  
#######################################

#######################################
# Predict the stock price of nth day
# based on historical data
# using Linear Regression
#######################################

# stock quotes
import quandl

# data frame processing
import pandas as pd

import datetime as dt
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

# sci-kit
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression

# serialize data and save
import pickle

DOWNLOAD_DATA = False
TRAIN_MODEL = False 
LRG_MODEL = "./LinearRegressionModel.pickel"

# quandl api key
quandl.ApiConfig.api_key = 'PySd8mej_Zbav262nViB'

# data frame from quandl for google stock price and volume data
if DOWNLOAD_DATA:
    df = quandl.get('WIKI/GOOGL')
    df.to_csv("./google_stock_quandl.csv")
else:
    df = pd.read_csv("./google_stock_quandl.csv")

print ">> Quandl GOOGL Stock data:"
print df.head()

OPEN='Adj. Open'
HIGH='Adj. High'
LOW='Adj. Low'
CLOSE='Adj. Close'
VOL='Adj. Volume'
HL_PCT = 'HL_PCT'
CHANGE_PCT = 'CHANGE_PCT'
LABEL = 'LABEL'
PREDICT = 'PREDICT'

# direct column manipulation of quandl data
# Not sure what [[ ]] mean ??
# this is the data columns we are interested in from original
df = df[[OPEN, HIGH, LOW, CLOSE, VOL]]

# Calculate % changes
df[HL_PCT] = (df[HIGH] - df[CLOSE]) / df[CLOSE] * 100.0
df[CHANGE_PCT] = (df[CLOSE] - df[OPEN]) / df[OPEN] * 100.0

# This is data we are going to operate on
# multivariable Linear regression will use these as parameters
df = df[[CLOSE, HL_PCT, CHANGE_PCT, VOL]]
print ">> Features:"
print df.head()

# Can't work with NaN data, so replace it with minimum value
df.fillna(-99999, inplace=True)

# we want to predict what is the stock price on nth day using linear regression
# forecast_out is the nth day

# Here nth day is 3rd day after today
#forecast_out = int(math.ceil(0.0009*len(df)))
# 33rd day
forecast_out = int(math.ceil(0.01*len(df)))
print ">> Forecast day:", forecast_out

# Create new labels column with forcasted price of nth day
FORECAST = CLOSE
df[LABEL] = df[FORECAST].shift(-forecast_out)

print ">> With Labels:"
print df.head()

# now we have complete data set with current price and forcasted price
# lets make it into input, label format

# Drop labels, and only feature data in numpy array
X = np.array(df.drop([LABEL], 1))
# Remove last few rows for which we dont have forecast prices
X = X[:-forecast_out]

# ??????
X = preprocessing.scale(X)

# last few rows for which we dont have forecast prices
# we can use it to forcast the price
X_lately = X[-forecast_out:] 
#print X
#print X_lately

# At this point, last few rows are gonna be empty, 
# since we dont have any data for those??
df.dropna(inplace=True)
Y = np.array(df[LABEL])
print "Total Data:", len(X), len(Y)

# Split data in train and test sets
# Cross validation, takes data, shuffles it and creates 2 sets train and test-20%
x_train, x_test, y_train, y_test = cross_validation.train_test_split(X, Y, test_size=0.2)
print "Train Data:", len(x_train), len(y_train)
print "Test Data:", len(x_test), len(y_test)

# we are not going to write linear regression manually
# will directly use scikit packages

if TRAIN_MODEL:
   # Linear regression 
   clf = LinearRegression() # Accuracy: 97%

   # SVM
   #clf = svm.SVR() # default kernel is linear regressions. Accuracy: 81%
   #clf = svm.SVR(kernel='poly') # Accuracy: 69%

   # Train the clf model
   clf.fit(x_train, y_train)
   with open(LRG_MODEL, 'wb') as fp:
       pickle.dump(clf, fp)
else:
   pickle_in = open(LRG_MODEL, 'rb')
   clf = pickle.load(pickle_in)
  
# Test and calculate accuracy
# formula for accuracy varies as per model defined in scikit
accuracy = clf.score(x_test, y_test)
#print y_test
print "Accuracy:", accuracy

# Lets predict
Y_predict = clf.predict(X_lately)
print Y_predict
#YY_predict = clf.predict([x_test[0]])
#print x_test[0], YY_predict

##################### Lets plot the results ###################
df[PREDICT] = np.nan
# in df we dont have dates, so lets use last_date as current date 
# and reclaculate all other dates in past
"""
last_date = df.iloc[-1].name
print last_date, type(last_date)
print type(last_date), last_date
last_unix = (pd.Timestamp(last_date) -  dt.datetime(1970,1,1))
print last_unix
one_day = 86400 # 24 * 60 * 60
next_unix = last_unix + one_day
"""
num = 0
for i in Y_predict:
    num += 1
    """next_date = dt.datetime.fromtimestamp(next_unix)
    next_unix += one_day"""
    nname = 'zzkau' + "-%s" % num
    df.loc[nname] = [np.nan for _ in range(len(df.columns)-1)] + [i]

print df.head()
print df.tail()

df[CLOSE].plot()
df[PREDICT].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
