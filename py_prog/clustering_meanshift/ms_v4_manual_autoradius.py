######################################
## Copyright 2017 @ Kau Gon  
######################################
# K-means clustering. Flat clustering
######################################
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.cluster import KMeans
# data generation
from sklearn.datasets.samples_generator import make_blobs

style.use('ggplot')

class MyMeanShift(object):
    def __init__(self, radius=None, radius_norm_step=100):
        # this ilike keamns iterations
        # we are using radius steps on max to work on radius
        self.bw = radius
        self.radius_norm_step = radius_norm_step

    def fit(self, train_data):
        self.data = train_data

        if self.bw is None:
            # since we dont have radius
            # take all data mean centroid to begin with
            all_data_centroid = np.average(self.data, axis=0)
            all_data_norm = np.linalg.norm(all_data_centroid)
            self.bw = all_data_norm / self.radius_norm_step 
        
        # which centroids to pickup to start with
        # each data_pt is a centroid
        self.centroids = {}
        for i in range(len(self.data)):
            #print self.data[i]
            self.centroids[i]= self.data[i]

        # weights???
        # 99,98..0
        weights = [i for i in range(self.radius_norm_step)][::-1]

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
                    distance = np.linalg.norm(featureset-centroid)
                    # data_pt is centroid. happens in first iteration
                    if distance == 0:
                        distance = 0.00000001
                    weight_index = int(distance/self.bw)
                    weight_index = min(weight_index, self.radius_norm_step-1)

                    to_add = (weights[weight_index]**2)*[featureset]
                    in_bw += to_add

                # mean of all data pts is new centroid of that cluster
                new_centroid = np.average(in_bw, axis=0)
                # array to tuple ??
                new_centroids.append(tuple(new_centroid))

            # reason to use tuple is to take set of it
            unique_centroids = sorted(list(set(new_centroids)))

            # too many centroids in the unique list
            # they ar every close to each other, so 
            # we are gonna get rid of those
            to_pop = []
            for i in unique_centroids:
                for ii in unique_centroids:
                    if i == ii:
                        pass
                    elif np.linalg.norm(np.array(i)-np.array(ii)) < self.bw:
                        # we need to delete this node
                        to_pop.append(ii)
                        break

            for i in to_pop:
                try:
                    unique_centroids.remove(i)
                except Exception as ex:
                    print ex
                    print i
                    pass

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

"""
x_list = [    [1, 2],
              [1.5, 1.8],
              [5, 8], 
              [8, 8],
              [1, 0.6],
              [9, 11],
              [8,2], [9,3], [10,2]
        ]
         
x = np.array(x_list)
"""
#plt.scatter(x[:,0], x[:, 1])
#plt.show()

# ????
x, y = make_blobs(n_samples=50, centers=3, n_features=2)

clf = MyMeanShift()
clf.fit(x)

print clf.centroids

colors = 20*["g", "r", "b", "k", "o"]

for centroid in clf.centroids: 
    #print centroid
    center = clf.centroids[centroid]
    #print center  
    plt.scatter(center[0], center[1], marker='x', s=150, color=colors[centroid])

# all exisisting point wrt their cluster
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
