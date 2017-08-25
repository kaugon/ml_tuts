######################################
## Copyright 2017 @ Kau Gon  
#######################################
# Linear regressions to find best fit line
# Use of SquaredError and r2 coefficient of determination
#
from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random

style.use('fivethirtyeight')

xs = np.array([1,2,3,4,5,6], dtype=np.float64)
ys = np.array([2,4,5,6,10,12], dtype=np.float64)

def create_dataset(size, variance, step=2, correlation=0):
    val = 1
    yl = []
    for i in range(size):
        y = val + random.randrange(-variance, variance)
        yl.append(y)
        if correlation:
            if correlation > 0:
                val += step
            else:
                val -= step
    xl = [i for i in range(len(yl))]
    return np.array(xl, dtype=np.float64), np.array(yl, dtype=np.float64)  

## Best fit line
def best_fit(xs, ys):
    mxs = mean(xs)
    m = (mxs * mean(ys)) - mean(xs*ys)
    m = m / ((mxs*mxs) - mean(xs*xs))
    b = mean(ys) - (m * mean(xs))
    return m, b

## Accuracy measurement: Squared Error
def sq_err(y_orig, y_bf):
    return sum((y_bf-y_orig)**2)

## Accuracy measurement: r2 coefficient
def coeff_r2(y_orig, y_bf):
    # mean y for every Y in origin y
    y_mean_line = [mean(y_orig) for y in y_orig]
    sq_err_bf = sq_err(y_orig, y_bf)
    sq_err_mean = sq_err(y_orig, y_mean_line)
    r2 = 1 - (sq_err_bf/sq_err_mean)
    return r2

xs, ys = create_dataset(size=40, step=100, step=2, correlation=1)

m,b = best_fit(xs, ys)
print "m: %s, b: %s" % (m, b)

regression_line = [m*x+b for x in xs]
predict_x = np.array([10, 13], dtype=np.float64)
predict_y = [m*x+b for x in predict_x]

r2 = coeff_r2(ys, regression_line)
print "r2: %s" % r2

## Plot
plt.scatter(xs,ys)
plt.scatter(predict_x,predict_y, s=100, color='r')
plt.plot(xs, regression_line)
plt.show()

