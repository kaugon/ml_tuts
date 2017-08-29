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

class MyMeanShift(object):
    # expriment with radius    
    def __init__(self, radius=4):
        # like kmeans, tolerance or iterations not required
        self.bw = radius

    def fit(self, train_data):
        self.data = train_data
        self.centroids = {}
        
        # which centroids to pickup to start with
        # each data_pt is a centroid
        for i in range(len(self.data)):
            #print self.data[i]
            self.centroids[i]= self.data[i]

        # optimize centroids
        while True:
            # new one t ocalculate
            new_centroids = []
            for i in self.centroids:
                in_bw = []
                centroid = self.centroids[i]
                # if data_pt is withn bw of centroid
                # maintain it in centroid-bw
                for featureset in self.data:
                    if np.linalg.norm(featureset-centroid) < self.bw:
                        in_bw.append(featureset)
                # mean of all data pts is new centroid of that cluster
                new_centroid = np.average(in_bw, axis=0)
                # array to tuple ??
                new_centroids.append(tuple(new_centroid))

            # reason to use tuple is to take set of it
            unique_centroids = sorted(list(set(new_centroids)))

            # compare previous with new unique centroids
            prev_centroids = dict(self.centroids)

            # store new ones again as np array
            self.centroids = {}
            for i in range(len(unique_centroids)):
                self.centroids[i] = np.array(unique_centroids[i])

            
            # Are we converging ?
            optimized = True
            for i in self.centroids:
                if not np.array_equal(self.centroids[i], prev_centroids[i]):
                    optimized = False
                    break

            # we are done. none of the centroids moved
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
              [9, 11],
              [8,2], [9,3], [10,2]
        ]
         
x = np.array(x_list)

plt.scatter(x[:,0], x[:, 1])
plt.show()

clf = MyMeanShift()
clf.fit(x)

print clf.centroids

#colors = ["g.", "r.", "b.", "k.", "o."]
colors = ["g", "r", "b", "k", "o"]

for centroid in clf.centroids: 
    #print centroid
    center = clf.centroids[centroid]
    #print center  
    plt.scatter(center[0], center[1], marker='x', s=150, color=colors[centroid])

for xx in x:
    cnum = clf.predict(xx)
    plt.scatter(xx[0], xx[1], color=colors[cnum])


plt.show()
exit(1)

x_predict_list = [[1,3], [8,9], [0,3], [5,4], [6,6]]
x_predict = np.array(x_predict_list)
for xx in x_predict:
    cnum = clf.predict(xx)
    plt.scatter(xx[0], xx[1], marker="*", color=colors[cnum])
plt.show()
