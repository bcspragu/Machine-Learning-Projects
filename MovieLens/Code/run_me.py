import numpy as np
import scipy.sparse
import kmeans
import json

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

def getRMSE(k):
  model = kmeans.kmeans(n_clusters=k)
  model.fit(Xtrain)

#Predict back the training ratings and compute the RMSE
  XtrainHat = model.predict(Xtrain,Xtrain)
  tr= model.rmse(Xtrain,XtrainHat)

#Predict the validation ratings and compute the RMSE
  XvalHat = model.predict(Xtrain,Xval)
  val= model.rmse(Xval,XvalHat)

  return (tr,val)

results = []
#Test k from 1 to 10
for k in range(1,11):
  results.append([])
  #Do 5 random restarts
  for runs in range(1,6):
    #Store the results
    results[k-1].append(getRMSE(k))

# Average, Max, and Min RMSE over k = 1 to 10 on training set
avg_tr = [np.mean([z[0] for z in y]) for y in results]
max_tr = [np.amax([z[0] for z in y]) for y in results]
min_tr = [np.amin([z[0] for z in y]) for y in results]

# Average, Max, and Min RMSE over k = 1 to 10 on validation set
avg_val = [np.mean([z[1] for z in y]) for y in results]
max_val = [np.amax([z[1] for z in y]) for y in results]
min_val = [np.amin([z[1] for z in y]) for y in results]

# Our actual model, with k=3
model = kmeans.kmeans(n_clusters=3)
model.fit(Xtrain)

clusters = model.cluster(Xtrain)

# Age, Gender, Occupation, and Address arrays for each cluster
resAge = [[],[],[]]
resGen = [[],[],[]]
resOcc = [[],[],[]]
resSt = [[],[],[]]

for i, x in enumerate(clusters):
    resAge[int(x)].append(Users[i][1])
    resGen[int(x)].append(Users[i][2])
    resOcc[int(x)].append(Users[i][3])
    resSt[int(x)].append(Users[i][4])

# 'zip.json' is a map from zip codes to states
with open('zip.json') as data_file:
    mapping = json.load(data_file)

for x in range(3):
    d = {}
    # Look at each zip code in the cluster and add it into our map
    for o in resSt[x]:
      if o in mapping:
        if mapping[o] in d:
            d[mapping[o]] += 1
        else:
            d[mapping[o]] = 1
      else:
        print("Couldn't find " + o)
    # Here, we'd build our pie chart

# centers is a k x 1682 array of ratings
centers = model.get_centers()

high = [list(reversed(sorted([(rating, Items[movie_id][1]) for movie_id, rating in enumerate(center)])))[:5] for center in centers]
low = [sorted([(rating, Items[movie_id][1]) for movie_id, rating in enumerate(center)])[:5] for center in centers]
