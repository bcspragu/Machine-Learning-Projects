import numpy as np
import time

from pyspark.conf import SparkConf
from pyspark.context import SparkContext

from operator import add
from math import sqrt
conf = SparkConf().setAppName("BSpragueHW3").setMaster("local[2]")
sc = SparkContext(conf=conf)

# Map the data to a tuple of (hour, (project code, page name), page views)
# We combine project code and page name with a delimeter of dash
def parse(i):
    def result(r):
        s = r.split()
        # The minus 5 accounts for the fact that we want to index our array
        # starting at one, not six
        return (i-5, (s[0], s[1]), int(s[2]))
    return result

def to_vector(r):
    # Create a numpy array of 18 elements
    n = np.zeros(18)
    # Set the array at the index of the number of hours minus 5 to the number
    # of page view, unless it's the target value, which we store separately
    target = 0
    if r[0] != 18:
        n[r[0]] = r[2]
    else:
        target = r[2]
    # Our new tuple is ((project code, page name), (18-element array with
    # arr[hour-6] set to page views, target value))
    # The feature vector and target value are wrapped in a tuple so they can be
    # added with one call to np.add
    return (r[1], (n.reshape(18,1), target))

def set_bias(r):
    # r[1] is our inner tuple, r[1][0] is the feature vector, r[1][0][0] is the
    # first term of the feature vector, which is the bias and should be 1
    r[1][0][0] = 1
    return r

def split_code_name(r):
    s = r[0]
    return (s[0], s[1], r[1][0], r[1][1])

def x_xtranspose(r):
    two_dim = r[2]
    return np.dot(two_dim, two_dim.T)

def xy_scale(r):
    return np.dot(r[2].T, r[3])

def predict(w):
    def result(r):
        return (r[0],r[1], np.dot(w.T, r[2])[0][0], r[3])
    return result

def even(r):
    return len(r[1]) % 2 == 0

def odd(r):
    return len(r[1]) % 2 == 1

def squared_diff(r):
    return (r[2]-r[3])**2

t1 = time.time()
# This one is for the server
#base = "/wikistates/{0}.txt"

# This one is for local testing
base = "/home/bsprague/Downloads/HW3Data/{0}.txt"

rdds = []
for i in range(6,24):
    f = base.format(i)
    rdd = sc.textFile(f)
    # We use our function-returing function to evade Spark's lazy evaluation
    rdd = rdd.map(parse(i))
    rdds.append(rdd)

# Combine all of our rdds
rdd = sc.union(rdds).filter(lambda r: r[1][0] == "en")

# We use our vector function from above
rdd = rdd.map(to_vector)

# We add all of the hours together, which is effectively adding a bunch of
# zeros and one page view count per column
rdd = rdd.reduceByKey(np.add)

# Set the bias term to 1
rdd = rdd.map(set_bias)

# Split the project code and project name out of the tuple we used earlier
rdd = rdd.map(split_code_name)

train = rdd.filter(even)
test = rdd.filter(odd)

nxxt = train.map(x_xtranspose)
nres = nxxt.reduce(np.add)
nxy = train.map(xy_scale)
nres2 = nxy.reduce(np.add)

nweights = np.dot(np.linalg.inv(nres), nres2.T)
pred = test.map(predict(nweights))

sq = pred.map(squared_diff)

summed = sq.reduce(add)
print(sqrt(summed/test.count()))
t2 = time.time()
print("Runtime:", t2-t1)
