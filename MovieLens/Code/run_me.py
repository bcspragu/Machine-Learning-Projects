import numpy as np
import scipy.sparse
import kmeans

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

#Train k-Means on the training data
k=2
model = kmeans.kmeans(n_clusters=k)
model.fit(Xtrain)

#Predict back the training ratings and compute the RMSE
XtrainHat = model.predict(Xtrain,Xtrain)
tr= model.rmse(Xtrain,XtrainHat)

#Predict the validation ratings and compute the RMSE
XvalHat = model.predict(Xtrain,Xval)
val= model.rmse(Xval,XvalHat)

#Predict the test ratings and compute the RMSE
XtestHat = model.predict(Xtrain,Xtest)
te= model.rmse(Xtest,XtestHat)

#Get the cluster assignments for the training data
z = model.cluster(Xtrain)
print(z)

#Get the clusters 
centers = model.get_centers()
print(centers)

print("K=%d Errors: %.7f %.7f %.7f"%(k,tr,val,te))
