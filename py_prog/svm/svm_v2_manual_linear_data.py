######################################
## Copyright 2017 @ Kau Gon  
######################################
# SVM custom implementation
# Predict: (X.W + b) > 0 ?
# Optimization:
#   Solve: Yi(Xi.W + b) >= 1
#           Minimiza W. Maximize b
######################################

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')


class MySVM(object):
    def __init__(self, visualization=True):
        self.vis = visualization
        self.colors = {1: 'r', -1: 'g'}
        if self.vis:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1,1,1)

    def fit(self, x_train):
        self.data = x_train

        # optimize
        # store calculatated mag of W with exisisting W and b values
        # { || W || : {W, b} }
        opt_dict = {}

        # transforms for W global minima
        transforms = [ [1,1],
                  [-1,1],   
                  [-1,-1],   
                  [1,-1],
                ]

        # we need starting point for finding optima
        # i.e. max value in data set
        # what are the highest value in our sample data
        # W [max, max]
        # here we are finding max or min value within [x,y] as well
        all_data = []
        for yi in self.data:
            for featureset in self.data[yi]:
                for feature in featureset:
                    all_data.append(feature)
        self.feature_max = max(all_data)
        self.feature_min = min(all_data)

        # steps for global minima
        steps = [ self.feature_max * 0.1,
                  self.feature_max * 0.01,
                  # do we really need such small step,
                  # it becomes expesive on compute  
                  self.feature_max * 0.001,
                  #self.feature_max * 0.0001,
                ]

        # steps for b
        b_range = 5 
        b_step = 5

        # current optima
        #  why 10 ?
        c_optima = self.feature_max * 10

        # optimization steps
        for step in steps:
            # W init from current optima
            w = np.array([c_optima, c_optima])
            # convex optimization, so we know when to stop
            optimized = False
            while not optimized:
                # bias range  (-max, +max, step=stepsize)
                for b in np.arange(-1*(self.feature_max*b_range),
                                    self.feature_max*b_range,
                                    step*b_step):
                    # W transformations
                    for trans in transforms:
                        w_t =  w * trans
                        # ??
                        found_option = True
                        # Are the current weight and bias good to fit all data_pts
                        # if any of the mdoesnt fit, then throw the wt and bias values
                        # otherwise store them
                        # fit: solve optimization equation
                        # yi(xi.w + b) >= 1 
                        for grp in self.data:
                            for xi in self.data[grp]:
                                yi=grp
                                r_t = yi * (np.dot(np.array(xi), w_t) + b)
                                if not r_t >= 1:
                                    found_option = False
                                    break

                        if found_option:
                           # L2 i.e. magnitude
                           w_mag = np.linalg.norm(w_t)
                           opt_dict[w_mag] = (w_t, b)

                # if x cord is less than 0, then we dont need to do anything else
                # we are done with this step.
                if w[0] < 0:
                    optimized = True     
                    print "optimized a step %s" % step
                else:
                   # step thru W 
                   w = w - step     

            # get the lowest mag    
            norms = sorted(opt_dict.keys())
            self.w, self.b = opt_dict[norms[0]]

            # 2?
            # update current optima
            c_optima = self.w[0] + step*2
 
            print "> current classification:" 
            for grp in self.data:
                for xi in self.data[grp]:
                    yi=grp
                    r_t = yi * (np.dot(np.array(xi), self.w) + self.b)
                    print xi, ":", r_t, ": Grp:", yi 

    def score(self, x_test, y_test):
        pass

    def predict(self, x_predict):
        # Predict the class
        # (X.W + b) > 0 ?
        # np.dot and b are scalars
        result = np.dot(np.array(x_predict), self.w) + self.b
        # why np.isgn ? and instead use condition > 0 ?
        result = np.sign(result)
        print result 
        if result != 0 and self.vis:
            self.ax.scatter(x_predict[0], x_predict[1], s=100, 
                    marker='*', c=self.colors[int(result)])
        return result
    
    def visualize(self):
        for grp in self.data:
            for pt in self.data[grp]:
                self.ax.scatter(pt[0], pt[1], color=self.colors[grp])

        # hyperplane v = x.w+b
        # psv = 1
        # nsv = -1
        # decbound = 0
        # return y cord for given x cord, w, b, and v
        def hyperplane(x,w,b,v):
            #vv = np.dot(x, w) + b
            #return vv - v
            # ??
            return (-w[0]*x-b+v) / w[1]

        # min and maximum x cord for the hyperplane
        hyp_x_min, hyp_x_max = (self.feature_min, self.feature_max)

        # w.x+b = 1
        # positive support vector hyperplane
        psv_min = hyperplane(hyp_x_min, self.w, self.b, 1)
        psv_max = hyperplane(hyp_x_max, self.w, self.b, 1)
        self.ax.plot([hyp_x_min, hyp_x_max], [psv_min, psv_max], 'k')

        # w.x+b = -1
        # positive support vector hyperplane
        nsv_min = hyperplane(hyp_x_min, self.w, self.b, -1)
        nsv_max = hyperplane(hyp_x_max, self.w, self.b, -1)
        self.ax.plot([hyp_x_min, hyp_x_max], [nsv_min, nsv_max], 'k')

        # w.x+b = 0
        # positive support vector hyperplane
        db_min = hyperplane(hyp_x_min, self.w, self.b, 0)
        db_max = hyperplane(hyp_x_max, self.w, self.b, 0)
        self.ax.plot([hyp_x_min, hyp_x_max], [db_min, db_max], 'y--')


        plt.show()
 
data_dict = { -1 : np.array([ [1,6], [2,8], [3,9] ]),
               1 : np.array([ [5,1], [6,-1], [8,3] ]),
            }


svm = MySVM()
svm.fit(data_dict)

x_predict = [ [1,10], [1,3], [3,4], [3,5], [5,5], [5,6], [7,1], [10, 2]]
for x_pt in x_predict:                
    svm.predict(x_pt)

svm.visualize()

