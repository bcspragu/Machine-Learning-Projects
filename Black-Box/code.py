import itertools
import operator

import numpy as np
from sklearn import cross_validation
from sklearn import svm
from sklearn import tree
from sklearn import neighbors
from sklearn import ensemble

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


train = np.load('train.npy')
# Remove the labels
test = np.load('test_distribute.npy')[:,1:]

data = train[:,1:]
target = train[:,0]

trimmer = SelectKBest(chi2, k=400).fit(data, target)

trimmed_data = trimmer.transform(data)
trimmed_test = trimmer.transform(test)

clf = ensemble.RandomForestClassifier(n_estimators=50)
pred = clf.fit(trimmed_data, target).predict(trimmed_test)

f = open('predictions.csv', 'w')
f.write("ID,Category\n")

for i, res in enumerate(pred):
    f.write("%d,%d\n" % (i+1,res))

f.close()


# I use the following code to find good hyperparameter values
#scores = cross_validation.cross_val_score(
    #clf, trimmed_data, target, cv=5)
#print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
