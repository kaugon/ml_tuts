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

x = np.array( [[1, 2],
              [1.5, 1.8],
              [5, 8], 
              [8, 8],
              [1, 0.6],
              [9, 11]]
            )

#plt.scatter(x[:,0],x[:,1])
#plt.show()

# expriment with number of clusters
clf = KMeans(n_clusters=2)
clf.fit(x)

# attributes of kemans cluster
centroids = clf.cluster_centers_
labels = clf.labels_

colors = ["g.", "r.", "b.", "k.", "o."]

for i in range(len(x)):
    plt.plot(x[i][0], x[i][1], colors[labels[i]], markersize=10)

print centroids
plt.scatter(centroids[:,0], centroids[:,1], marker='x', s=150)
plt.show()

