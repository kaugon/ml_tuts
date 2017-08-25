import numpy as np
from sklearn import preprocessing, cross_validation, neighbors
import pandas as pd

# dataset from: 
# http://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Original%29
# Add header below
df = pd.read_csv('./data/breast-cancer-wisconsin.data')
t_id = 'id'
t_class = 'class'
# id, clump_thickness, unif_cell_size, unif_cell_shape, marginal_adhesion, single_cell_size, bare_neuclei, bland_chromatin, normal_neuclei, mitoses, class  
# 1000025,5,1,1,1,2,1,3,1,1,2

# dataset.name: notes: missing data ?? 
# replacement, -99999 implies outlier. most algos recognize it
df.replace('?', -99999, inplace=True)
# no need for first column id, since it has nothing to do 
# with classification.
# if you keep this column and train, accuracy would go down signifincantly
# becasue its just random data for classification purpose
df.drop([t_id], 1, inplace=True)
print df.head()

# input/features without class results
x = np.array(df.drop([t_class],1))
# predict class
y = np.array(df[t_class])

x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size=0.2)

clf = neighbors.KNeighborsClassifier()
clf.fit(x_train, y_train)
accuracy = clf.score(x_test, y_test)
print "Accuracy:", accuracy

x_predict = np.array([[7,3,5,8,10,1,2,3,2],[7,3,1,2,3,4,5,6,2]])
# we may need to use reshape, if testing with only 1 sample
# with 2 smaples, 3d  shape is created fine
#x_predict = x_predict.reshape(1, -1)
print x_train.shape, len(x_train[0]), x_predict.shape, len(x_predict[0])
y_predict = clf.predict(x_predict)
print "predict:", x_predict, "result:", y_predict

