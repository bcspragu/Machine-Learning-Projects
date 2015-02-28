import itertools
import time

import numpy as np
from sklearn import svm
from sklearn import neighbors
from sklearn import tree

train = np.load('train.npy')
# Remove the labels
test = np.load('test_distribute.npy')[:,1:]

data = train[:,1:]
target = train[:,0]

def bench(name, clf):
    print "Training", name
    t0 = time.time()
    clf.fit(data, target)
    t1 = time.time()
    print "Training", name, "took", t1-t0, "s"

    print "Classifying", name
    t2 = time.time()
    pred = clf.predict(test)
    t3 = time.time()
    print "Classifying", name, "took", t3-t2, "s"

    print "Writing results"
    f = open('predictions-' + name + '.csv', 'w')
    f.write("ID,Category\n")

    for i, res in enumerate(pred):
        f.write("%d,%d\n" % (i+1,res))

    f.close()

bench('SVM', svm.SVC())
bench('DTr', tree.DecisionTreeClassifier())
bench('KNN', neighbors.KNeighborsClassifier())
