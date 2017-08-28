######################################
## Copyright 2017 @ Kau Gon  
######################################
# K-means clustering. Flat clustering
######################################
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.cluster import KMeans

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
