import itertools
import operator

import numpy as np
from sklearn import cross_validation
from sklearn import neighbors


train = np.load('train.npy')
# Remove the labels
test = np.load('test_distribute.npy')[:,1:]

data = train[:,1:]
target = train[:,0]

np.set_printoptions(threshold='nan')
print target
#print neighbors.KNeighborsClassifier(n_neighbors=1).fit(data, target).predict(data)

# I use the following code to find good hyperparameter values
#scores = cross_validation.cross_val_score(
    #clf, data, target, cv=5)
#print("Accuracy: %f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
