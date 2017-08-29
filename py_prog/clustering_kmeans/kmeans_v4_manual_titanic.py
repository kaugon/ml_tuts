######################################
## Copyright 2017 @ Kau Gon  
######################################
# merged v2 and v3 versions together
# v2: Titanic data with scikit kmeans
# v3: our own impl of kmeans 
# here using our kmeans implementation on titanic data
######################################
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.cluster import KMeans
import pandas as pd
from sklearn import preprocessing, cross_validation

style.use('ggplot')

class MyKMeans(object):
    def __init__(self, k=2, tol=0.001, max_iter=300):
        # tolerance is how much centroid moves
        # stop at these many iterations
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, train_data):
        self.data = train_data
        self.centroids = {}
        
        # which centroids to pickup to start with
        # pickup first k from dataset
        for i in range(self.k):
            print self.data[i]
            self.centroids[i]= self.data[i]

        # optimize centroids
        for niter in range(self.max_iter):
            #print "> centroids: ", self.centroids
            # k cluster
            # keys will be centroids and list of values
            # since centroids are gonna move per iteration to be accurate
            # cluster would change/re-initialized
            self.clusters = {}
            for i in range(self.k):
                self.clusters[i] = []

            for featureset in self.data:
                distance = []
                for center in self.centroids:
                    distance.append(np.linalg.norm(featureset - self.centroids[center]))
                # argmin ?
                #print featureset, distance
                predict_cluster = distance.index(min(distance))
                self.clusters[predict_cluster].append(featureset)
                
            # at this point, all data points are clustered
            #print "> clusters: ", self.clusters
            prev_centroids = dict(self.centroids)
            # note: if we do:
            #  prev = self.centroids
            # dictionary is not replicated. its just new name for same dict
            # so if dict gets updated, prev_centroids will be updated as well
            # we dont want that hence,
            # prev = dict(self.centroids)
           
            # find the mean for each cluster
            for cluster in self.clusters:
                # each one is list of points in that cluster
                # take its average, its new cord i.e. new centroid for that cluster
                cavg = np.average(self.clusters[cluster], axis=0)
                #print "cavg: ", cluster, cavg   
                self.centroids[cluster] = np.average(self.clusters[cluster], axis=0)

            optimized = True

            # check tolerance with previous centroids
            for center in self.centroids:
                prev = prev_centroids[center]
                curr = self.centroids[center]
                #?? sum 
                if np.sum( (curr-prev)/prev*100.0) > self.tol:
                    print np.sum( (curr-prev)/prev*100.0)
                    optimized = False

            if optimized:
                break 

    def predict(self, test_data):
        # how far is this test_pt from centroids
        distance = []
        for center in self.centroids:
            distance.append(np.linalg.norm(test_data - self.centroids[center]))
            # argmin ?
            predict_cluster = distance.index(min(distance))
        return predict_cluster


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
clf = MyKMeans(k=2)
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
    if y_predict == Y[i]:
        correct += 1

print "Accuracy:", (correct/len(X))


"""
x_list = [    [1, 2],
              [1.5, 1.8],
              [5, 8], 
              [8, 8],
              [1, 0.6],
              [9, 11]
        ]
x_predict_list = [[1,3], [8,9], [0,3], [5,4], [6,6]]
         
x = np.array(x_list)

clf = MyKMeans(k=2)
clf.fit(x)

#colors = ["g.", "r.", "b.", "k.", "o."]
colors = ["g", "r", "b", "k", "o"]

for centroid in clf.centroids: 
    #print centroid
    center = clf.centroids[centroid]
    #print center  
    plt.scatter(center[0], center[1], marker='x', s=150, color=colors[centroid])

for cluster in clf.clusters:
    color = colors[cluster]+'.'
    for pt in clf.clusters[cluster]:
        plt.plot(pt[0], pt[1], color)

x_predict = np.array(x_predict_list)
for xx in x_predict:
    cnum = clf.predict(xx)
    plt.scatter(xx[0], xx[1], marker="*", color=colors[cnum])
plt.show()
"""
