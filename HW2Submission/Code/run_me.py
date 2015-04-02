import datetime
import numpy as np

from math import exp
from math import log

from sklearn.cross_validation import KFold
from sklearn.metrics import mean_absolute_error

from sklearn import ensemble
from sklearn import feature_selection
from sklearn import linear_model
from sklearn import neighbors
from sklearn import preprocessing
from sklearn import svm

# Write out our predictions line-by-line and save it to a timestamped file
def FireWritePredictions(pred):
    time_str = datetime.datetime.now().strftime("%d-%m-%y-%H:%M:%S")
    f = open('../Predictions/ForestFires/' + time_str + 'prediction.csv', 'w')
    f.write("ID,Target\n")

    for i, res in enumerate(pred):
        f.write("%0.1f,%f\n" % (float(i+1),res))

    f.close()

# The pipeline for the forest fire data first uses the OneHotEncoder on the
# month and day features of the dataset. Then it scales all of the values to be
# between 0 and 1, because they have strange ranges. After normalizing this
# data, we run SVR on the processed data
def FirePipeline(train, test):
    preds = []
    
    tr_data = train[:,1:]
    target = train[:,0]
    test = test[:,1:]

    ss = preprocessing.OneHotEncoder(categorical_features=[2,3])
    ss.fit(tr_data, target)
    tr_data = ss.transform(tr_data)
    test = ss.transform(test)

    ss = preprocessing.MinMaxScaler()
    ss.fit(tr_data.toarray(), target)
    tr_data = ss.transform(tr_data.toarray())
    test = ss.transform(test.toarray())

    tr_data = preprocessing.normalize(tr_data)
    test = preprocessing.normalize(test)

    clf = svm.SVR()
    clf.fit(tr_data, target)

    pred = clf.predict(test)
    return pred


# Mean Absolute Error
def ScoreMAE(actual, pred):
    return mean_absolute_error(actual, pred)

# To cross-validate a regressor, we pass in the regressor, the data, and the
# number of sets we'd like to break our data into. We get the default params of
# the classifier, to reset after each train/test cycle. We average over our
# folds different runs
def FireCrossValClf(clf, data, folds):
    params = clf.get_params()
    kf = KFold(len(data), n_folds=5, shuffle=True, random_state=0)
    scores = []
    for train, test in kf:
        tr_data, te_data = data[train], data[test]
        clf.fit(tr_data[:,1:], tr_data[:,0])
        pred = clf.predict(te_data[:,1:])
        clf.set_params(**params)
        scores.append(ScoreMAE(te_data[:,0], pred))
    return np.average(scores)

# Run the cross-validation process on our above-defined pipeline
def FireCrossValPipeline(data, folds):
    kf = KFold(len(data), n_folds=5, shuffle=True, random_state=0)
    scores = []
    for train, test in kf:
        tr_data, te_data = data[train], data[test]
        pred = Pipeline(tr_data, te_data)
        scores.append(ScoreMAE(te_data[:,0], pred))
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


# Write out our predictions line-by-line and save it to a timestamped file
def CrimeWritePredictions(pred):
    time_str = datetime.datetime.now().strftime("%d-%m-%y-%H:%M:%S")
    f = open('../Predictions/CommunityCrime/' + time_str + 'prediction.csv', 'w')
    f.write("ID,Target\n")

    for i, res in enumerate(pred):
        f.write("%0.1f,%f\n" % (float(i+1),res))

    f.close()

# Preprocessing steps didn't help this pipeline, so this pipeline uses optimal
# hyperparameters for different classifiers using the above method, and takes a
# straight average of them.
def CrimePipeline(train, test):
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


# Root Mean Squared Error
def ScoreRMSE(actual, pred):
    return sqrt(mean_squared_error(actual, pred))

# To cross-validate a regressor, we pass in the regressor, the data, and the
# number of sets we'd like to break our data into. We get the default params of
# the classifier, to reset after each train/test cycle. We average over our
# folds different runs
def CrimeCrossValClf(clf, data, folds):
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

# Run the cross-validation process on our above-defined pipeline
def CrimeCrossValPipeline(data, folds):
    kf = KFold(len(data), n_folds=5, shuffle=True, random_state=0)
    scores = []
    for train, test in kf:
        tr_data, te_data = data[train], data[test]
        pred = CrimePipeline(tr_data, te_data)
        scores.append(ScoreRMSE(te_data[:,0], pred))
    return np.average(scores)

crime_train = np.load('crime_train.npy')
crime_test = np.load('crime_test_distribute.npy')

fire_train = np.load('fire_train.npy')
fire_test = np.load('fire_test_distribute.npy')

pred = CrimePipeline(crime_train, crime_test)
CrimeWritePredictions(pred)

pred = FirePipeline(fire_train, fire_test)
FireWritePredictions(pred)
