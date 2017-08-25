######################################
## Copyright 2017 @ Kau Gon  
#######################################
# Linear regressions to find best fit line
#
from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

style.use('fivethirtyeight')

xs = np.array([1,2,3,4,5,6], dtype=np.float64)
ys = np.array([2,4,5,6,10,12], dtype=np.float64)

def best_fit(xs, ys):
    mxs = mean(xs)
    m = (mxs * mean(ys)) - mean(xs*ys)
    m = m / ((mxs*mxs) - mean(xs*xs))
    b = mean(ys) - (m * mean(xs))
    return m, b

m,b = best_fit(xs, ys)
print m, b

regression_line = [m*x+b for x in xs]
predict_x = np.array([10, 13], dtype=np.float64)
predict_y = [m*x+b for x in predict_x]

plt.scatter(xs,ys)
plt.scatter(predict_x,predict_y, color='r')
plt.plot(xs, regression_line)
plt.show()

