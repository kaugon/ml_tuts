######################################
## Copyright 2017 @ Kau Gon  
######################################
# K-means clustering non-numeric data.
######################################
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.cluster import KMeans, MeanShift
import pandas as pd
from sklearn import preprocessing, cross_validation

style.use('ggplot')

df = pd.read_excel('../data/titanic.xls')
"""
pclass
survived
name
sex
age
sibsp - siblings/children aboard
parch - parents/children aboard
ticket
fare
cabin
embarked
boat
body
home.dest
"""
print "> original:\n", df.head()
original_df = pd.DataFrame.copy(df)

# expriment with which columns make impact on accuracy.
df.drop(['name', 'body'],1, inplace=True)
print "> Drops:\n", df.head()

# ??
df.convert_objects(convert_numeric=True)
df.fillna(0, inplace=True)
print "> Cleanup:\n", df.head()

# create dict of all non_numeric data
# replace those with unique numbers (x)
def handle_non_numeric_data(df):
    columns = df.columns.values
    for col in columns:
        # {"Female": 0 } etc.
        text_digit_vals = {}

        def convert_to_int(val):
            return text_digit_vals[val]

        # non_numeric data
        if not df[col].dtype in [np.int64, np.float]:
            # unique nonnumeric values for that column
            col_vals = set(df[col].values.tolist())
            # unique numbers in given columns
            x = 0
            # create numerical values for those type
            for unique in col_vals:
                if not unique in text_digit_vals:
                    text_digit_vals[unique] = x
                    x += 1

            # replace the column
            df[col] = list(map(convert_to_int, df[col])) 

    return df

df = handle_non_numeric_data(df)
print "> Numeric:\n", df.head()

# Determine who survived
X = np.array(df.drop(['survived'], 1).astype('float'))

# preprocessing increases the accuracy 
# ???
X = preprocessing.scale(X)

Y = np.array(df['survived'])

#clf = KMeans(n_clusters=2)
clf = MeanShift()
clf.fit(X)

# in given clusters, is our prediction correct
correct = float(0)
for i in range(len(X)):
    x_predict = np.array(X[i])
    x_predict = x_predict.reshape(-1, len(x_predict))
    y_predict = clf.predict(x_predict)
    # cluster number wrt surivived ??? thats not ocrrect mapping
    # so results might vary depending on the cluster number
    # i.e. cl#0 = survived#1
    # or cl#0 = survived#0
    # based on accuracy we will know.
    if y_predict[0] == Y[i]:
        correct += 1

print "Accuracy:", (correct/len(X))

labels = clf.labels_
cluster_centers = clf.cluster_centers_

# which cluster the sample is in ?
original_df['cluster_group'] = np.nan
for i in range(len(X)):
    original_df['cluster_group'].iloc[i] = labels[i]

# cluster_id to survival rate
survival_rates = {}
n_clusters = len(np.unique(labels))
print n_clusters
for i in range(n_clusters):
    temp_df = original_df[ (original_df['cluster_group']==float(i)) ]
    survival_cluster = temp_df[ (temp_df['survived']==1) ]
    survival_rate = len(survival_cluster)/float(len(temp_df))
    survival_rates[i] = survival_rate 

print survival_rates
