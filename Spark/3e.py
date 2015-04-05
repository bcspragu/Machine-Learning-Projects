from operator import add
from math import sqrt

def squared_diff(r):
    return (r[2]-r[3])**2

sq = pred.map(squared_diff)
summed = sq.reduce(add)

rmse = sqrt(summed/test.count())
