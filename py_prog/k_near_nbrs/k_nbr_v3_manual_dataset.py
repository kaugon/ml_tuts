import numpy as np
from math import sqrt
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import random

# will not use scikit

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
    confidence = float(vote_counts[0][1]) / k 
    #print k_nbrs
    #print votes
    #print vote_counts
    #print vote_grp
    return vote_grp, confidence

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
full_data = df.astype(float).values.tolist()
print full_data[:10]

# shuffled data
random.shuffle(full_data)

# train vs test
test_size = int(0.3*len(full_data))
train_data = full_data[:-test_size]
test_data = full_data[-test_size:]

# build dictionary of cluster/class and samples
# dict keys are 2 and 4. the class numbers from the data, with empty sample list
# 
train_dict = {2:[], 4:[]}
test_dict = {2:[], 4:[]}

# -1 is index of class in sample data, using it as key
for sample in train_data:
    train_dict[sample[-1]].append(sample[:-1])
for sample in test_data:
    test_dict[sample[-1]].append(sample[:-1])

#print test_dict.keys()
# from test data set, we know the answer/class
# so test the classifier
correct_results = float(0)
total_samples = 0
for t_group in test_dict:
    for t_sample in test_dict[t_group]:
        vote_group, confidence = k_near_nbr(train_dict, t_sample, k=25)
        if vote_group == t_group:
            correct_results += 1
        else:
            print confidence
        total_samples += 1

accuracy = correct_results / total_samples
print "Accuracy: ", accuracy

"""
style.use('ggplot')
#style.use('fivethirtyeight')
for group in dataset:
    for p in dataset[group]:
        plt.scatter(p[0], p[1], s=100, color=group)

plt.scatter(test_pt[0], test_pt[1], s=300, color=vote_grp)
plt.show()
""" 
