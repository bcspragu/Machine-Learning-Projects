import itertools
import operator

import numpy as np
from sklearn import cross_validation
from sklearn import feature_selection
from sklearn.ensemble import RandomForestClassifier

train = np.load('train.npy')
# Remove the labels
test = np.load('test_distribute.npy')[:,1:]

data = train[:,1:]
target = train[:,0]

trimmer = feature_selection.SelectPercentile(percentile=15).fit(data, target)
data = trimmer.transform(data)
test = trimmer.transform(test)

pred = RandomForestClassifier(n_estimators=100).fit(data, target).predict(test)

f = open('predictions.csv', 'w')
f.write("ID,Category\n")

for i, res in enumerate(pred):
    f.write("%d,%d\n" % (i+1,res))

f.close()
# I use the following code to find good hyperparameter values
#scores = cross_validation.cross_val_score(
    #clf, data, target, cv=5)
#print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
