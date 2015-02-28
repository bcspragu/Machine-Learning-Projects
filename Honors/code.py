import itertools
import operator

import numpy as np
from sklearn import cross_validation
from sklearn import feature_selection
from sklearn import ensemble
from sklearn import neighbors
from sklearn import svm

train = np.load('train.npy')[:1000]
# Remove the labels
test = np.load('test.npy')[:1000]


train_data = train[:,1:]
target = train[:,0]

test_data = test[:,1:]
names = test[:,0]

def most_common(L):
  # get an iterable of (item, iterable) pairs
  SL = sorted((x, i) for i, x in enumerate(L))
  groups = itertools.groupby(SL, key=operator.itemgetter(0))
  def _auxfun(g):
    item, iterable = g
    count = 0
    min_index = len(L)
    for _, where in iterable:
      count += 1
      min_index = min(min_index, where)
    return count, -min_index
  # pick the highest-count/earliest item
  return max(groups, key=_auxfun)[0]

# Now I rock out some ensemble-ception, because hopefully these are all
# wrong in different ways, and I choose the most common answer. If they're all
# different let's go with the first, and I'll order my classifiers based on
# which one is my favorite
def predictificate(data, target, test, clfs):
    res = []
    for clf in clfs:
        clf.fit(data, target)
        res.append(clf.predict(test))

    pred = [most_common(x) for x in zip(*res)]
    f = open('final-predictions.csv', 'w')
    f.write("ID,Category\n")

    for i, res in enumerate(pred):
        f.write("%d,%d\n" % (i+1,res))

    f.close()

clfs = []

# Through cv testing, I found the optimal number of estimators to be 15
clfs.append(ensemble.RandomForestClassifier(n_estimators=150))
clfs.append(ensemble.GradientBoostingClassifier(n_estimators=200))
clfs.append(ensemble.AdaBoostClassifier(n_estimators=135))
#clfs.append(neighbors.KNeighborsClassifier(n_neighbors=10))
#clfs.append(svm.SVC())

#predictificate(data, target, test, clfs)

clf = ensemble.RandomForestRegressor()
clf.fit(train_data, target)
probabilities = clf.predict(test_data)
print probabilities
#f = open('preds.csv', 'w')

#f.write('image,'+ names.join(','))
# I use the following code to find good hyperparameter values
