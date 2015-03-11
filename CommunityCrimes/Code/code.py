import datetime

import numpy as np
from math import sqrt

from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error

from sklearn import ensemble
from sklearn import feature_selection
from sklearn import linear_model
from sklearn import neighbors
from sklearn import svm

def WritePredictions(pred):
    time_str = datetime.datetime.now().strftime("%d-%m-%y-%H:%M:%S")
    f = open('../Predictions/' + time_str + 'prediction.csv', 'w')
    f.write("ID,Target\n")

    for i, res in enumerate(pred):
        f.write("%0.1f,%f\n" % (float(i+1),res))

    f.close()

def Pipeline(train, test):
    preds = []
    
    tr_data = train[:,1:]
    target = train[:,0]
    test = test[:,1:]
    
    clf = ensemble.RandomForestRegressor(n_estimators=101, random_state=0)
    clf.fit(tr_data, target)
    preds.append(clf.predict(test))

    clf = linear_model.LassoLars(alpha=0.0002)
    clf.fit(tr_data, target)
    preds.append(clf.predict(test))

    clf = linear_model.ElasticNet(alpha=0.002, l1_ratio=0.6) 
    clf.fit(tr_data, target)
    preds.append(clf.predict(test))

    clf = linear_model.BayesianRidge()
    clf.fit(tr_data, target)
    preds.append(clf.predict(test))

    return np.mean( np.array(preds), axis=0 )


def ScoreRMSE(actual, pred):
    return sqrt(mean_squared_error(actual, pred))

def CrossValClf(clf, data, folds):
    params = clf.get_params()
    kf = KFold(len(data), n_folds=5, shuffle=True, random_state=0)
    scores = []
    for train, test in kf:
        tr_data, te_data = data[train], data[test]
        clf.fit(tr_data[:,1:], tr_data[:,0])
        pred = clf.predict(te_data[:,1:])
        clf.set_params(**params)
        scores.append(ScoreRMSE(te_data[:,0], pred))
    return np.average(scores)

def CrossValPipeline(data, folds):
    kf = KFold(len(data), n_folds=5, shuffle=True, random_state=0)
    scores = []
    for train, test in kf:
        tr_data, te_data = data[train], data[test]
        pred = Pipeline(tr_data, te_data)
        scores.append(ScoreRMSE(te_data[:,0], pred))
    return np.average(scores)

# This method will take an array of values for a set of hyperparameters and
# return a tupe containing the the best hyperparameter combination and its
# cross-validated score. It recursively selects parameters, keeping track of
# the ones it has already tried.

# Here, kwargs should be specified as a dictionary from hyperparameter name, to
# an array of values it can take. For example, calling HyperparameterSelection
# on KNN trying to set distance metric and k (n_neighbors) would look like:

# HyperparameterSelection(KNeighborsRegressor, data, {'metric': ['minkowski', 'chebychev'], 'n_neighbors': range(1,10)})
# HyperparameterSelection will return the combination that minimizes the RMSE
def HyperparameterSelection(clf, data, **kwargs):
    # Should really fix this
    best_score = 100000
    best_args = kwargs

    # We make a list of parameters that still haven't been searched over
    unexplored = [k for (k,v) in kwargs.iteritems() if isinstance(v, list)]

    # We've already explored all values
    if len(unexplored) == 0:
        score = CrossValClf(clf(**kwargs), data, 5)
        return (kwargs, score)

    # We search over the first parameter we haven't searched yet
    explore_param = unexplored[0]
    for val in kwargs[explore_param]:
        kwargs[explore_param] = val
        params, score = HyperparameterSelection(clf, data, **kwargs)
        if score < best_score:
            best_score = score
            best_args = params

    return (best_args, best_score)

train = np.load('train.npy')
test = np.load('test_distribute.npy')

# Best result is 0.18800337317639737 with n_neighbors = 19, metric = manhattan, weights = uniform
#print HyperparameterSelection(neighbors.KNeighborsRegressor, train, metric=['minkowski', 'manhattan'], n_neighbors=range(15, 25), weights=['uniform', 'distance'])

# Best result is 0.13776738841510364 with n_estimators = 101
#print HyperparameterSelection(ensemble.RandomForestRegressor, train, n_estimators=range(100, 105), random_state=0)

# Best result is 0.17740608939889385 with nu=1
#print HyperparameterSelection(svm.NuSVR, train, nu=[10**x for x in range(-5,5)], random_state=0) 

# Best result is 0.13896059666185639 with alpha=0.0002
#print HyperparameterSelection(linear_model.LassoLars, train, alpha=[y*10**x for x in range (-4,-2) for y in range (1,9)]) 

# Best result is 0.13818134707029511 with alpha=0.002, l1_ratio=0.6
#print HyperparameterSelection(linear_model.ElasticNet, train, alpha=[y*10**x for x in range (-4,-2) for y in range (1,9)], l1_ratio=[x*0.1 for x in range(0,11)]) 

# Best result didn't change by altering parameters, was 0.13890488467827974
#print HyperparameterSelection(linear_model.BayesianRidge, train)

print CrossValPipeline(train, 5)
#pred = Pipeline(train, test)
#WritePredictions(pred)
