import itertools
import operator

import numpy as np
from sklearn import cross_validation
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

train = np.load('train.npy')
# Remove the labels
test = np.load('test_distribute.npy')[:,1:]

data = train[:,1:]
target = train[:,0]

# Originally, I thought removing these fields would be good, but I actually do better by not removing them, tragic
#data = np.delete(data, np.s_[[x for x in range(990) if x % 15 in [3,4,5,10,11,12,13,14]]], 1)
#test = np.delete(test, np.s_[[x for x in range(990) if x % 15 in [3,4,5,10,11,12,13,14]]], 1)

#clf = svm.LinearSVC()
pred = RandomForestClassifier(n_estimators=100).fit(data, target).predict(test)

f = open('predictions.csv', 'w')
f.write("ID,Category\n")

for i, res in enumerate(pred):
    f.write("%d,%d\n" % (i+1,res))

f.close()
