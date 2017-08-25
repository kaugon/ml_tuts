import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from matplotlib import style
from collections import Counter

# L2 distance for points in (x,y) repr
def dist_l2(p1, p2):
    return sqrt( (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 )

# L2 distance for multivariate points
def dist_l2_np(np1, np2):
    return np.sqrt(np.sum((np1-np2)**2))

def k_near_nbr(data, predict, k=3):
    if len(data) >= k:
        print "Invalid value of K or data"  

    # algo
    l2_list = []
    for group in data:
        for pt in data[group]:
            #l2 = dist_l2(pt, predict)
            #l2 = dist_l2_np(np.array(pt), np.array(predict))
            l2 = np.linalg.norm(np.array(pt)-np.array(predict))
            l2_list.append([l2, group])

    # lowest distance from k neighbors
    k_nbrs = sorted(l2_list)[:k]
    # k nbr votes
    votes = [nbr[1] for nbr in k_nbrs]
    # count per group
    vote_counts = Counter(votes).most_common()
    vote_grp = vote_counts[0][0] 
    print k_nbrs
    print votes
    print vote_counts
    print vote_grp
    return vote_grp

dataset = {'k':[[1,2], [2,3], [3,1]],
           'y':[[6,5], [4,3], [5,7]],
          }

test_pt = [3,6]

vote_grp = k_near_nbr(dataset, test_pt)

style.use('ggplot')
#style.use('fivethirtyeight')
for group in dataset:
    for p in dataset[group]:
        plt.scatter(p[0], p[1], s=100, color=group)

plt.scatter(test_pt[0], test_pt[1], s=300, color=vote_grp)
plt.show()
 
