import itertools
import operator

import numpy as np
from sklearn import cross_validation
from sklearn import tree
from sklearn import naive_bayes
from sklearn import ensemble
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier

# I stole this gem of a function from: http://stackoverflow.com/questions/1518522/python-most-common-element-in-a-list
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
    f = open('predictions.csv', 'w')
    f.write("ID,Category\n")

    for i, res in enumerate(pred):
        f.write("%d,%d\n" % (i+1,res))

    f.close()

train = np.load('train.npy')
# Remove the labels
test = np.load('test_distribute.npy')[:,1:]

data = train[:,1:]
target = train[:,0]

clfs = []
# Through cv testing, I found optimal C to be 0.0001
clfs.append(naive_bayes.MultinomialNB(alpha=10**-5))

# Through cv testing, I found optimal depth to be 11
clfs.append(tree.DecisionTreeClassifier(max_depth=11))

# Through cv testing, I found the optimal number of estimators to be 15
clfs.append(ensemble.RandomForestClassifier(n_estimators=40))

# You know the deal, for the next two, messy testing to get hyperparameters
clfs.append(BaggingClassifier(KNeighborsClassifier(),max_samples=0.6, max_features=0.3))
clfs.append(AdaBoostClassifier(n_estimators=40))

predictificate(data, target, test, clfs)

# I use the following code to find good hyperparameter values
#scores = cross_validation.cross_val_score(
    #clf, data, target, cv=5)
#print("Accuracy: %0.2f (+/- %0.2f), %d" % (scores.mean(), scores.std() * 2, x))
