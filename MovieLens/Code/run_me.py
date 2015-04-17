import numpy as np
import scipy.sparse
import kmeans
import matplotlib.pyplot as plt

#Make sure we get consistent, reproducible results
np.random.seed(seed=1)
#Define the data directory (change if you place data elsewhere)
data_dir = "/home/bsprague/Projects/CS589/MovieLens/Data/" 

#Load the training ratings
A       = np.load(data_dir + "train.npy")
A.shape = (1,)
Xtrain  = A[0]

#Load the validation ratings
A       = np.load(data_dir + "validate.npy")
A.shape = (1,)
Xval    = A[0]

#Load the test ratings
A       = np.load(data_dir + "test.npy")
A.shape = (1,)
Xtest   = A[0]

#Load the user, item, and genre information
Users   = np.load(data_dir + "users.npy")
Items   = np.load(data_dir + "items.npy")
Genres  = np.load(data_dir + "genres.npy")

model = kmeans.kmeans(n_clusters=3)
model.fit(Xtrain)

#Predict back the training ratings and compute the RMSE
#XtrainHat = model.predict(Xtrain,Xtrain)
#tr= model.rmse(Xtrain,XtrainHat)

#Predict the validation ratings and compute the RMSE
#XvalHat = model.predict(Xtrain,Xval)
#val= model.rmse(Xval,XvalHat)

#Predict the test ratings and compute the RMSE
#XtestHat = model.predict(Xtrain,Xtest)
#te= model.rmse(Xtest,XtestHat)

clusters = model.cluster(Xtrain)
resAge = [[],[],[]]
for i, x in enumerate(clusters):
    resAge[int(x)].append(Users[i][3])

print(resAge)
