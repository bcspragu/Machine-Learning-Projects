from math import sqrt

import numpy as np
from sklearn import cross_validation
from sklearn import ensemble

def write_predictions(pred):
    f = open('prediction.csv', 'w')
    f.write("ID,Target\n")

    for i, res in enumerate(pred):
        f.write("%0.1f,%f\n" % (float(i+1),res))

    f.close()

train = np.load('train.npy')
test = np.load('test_distribute.npy')

data = train[:,1:]
target = train[:,0]

test_data = test[:,1:]

clf = ensemble.RandomForestRegressor()
clf.fit(data, target)
pred = clf.predict(test_data)
write_predictions(pred)

#score = sqrt(np.mean([-x for x in cross_validation.cross_val_score(clf, data, target, scoring='mean_squared_error')]))

#print score

