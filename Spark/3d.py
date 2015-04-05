def en_even(r):
    return r[0] == "en" and len(r[1]) % 2 == 0

def en_odd(r):
    return r[0] == "en" and len(r[1]) % 2 == 1

def predict(w):
    def result(r):
        return (r[0],r[1], np.dot(w.T, r[2])[0][0], r[3])
    return result

train = rdd.filter(en_even)
test = rdd.filter(en_odd)

nxxt = train.map(x_xtranspose)
nres = nxxt.reduce(np.add)

nxy = train.map(xy_scale)
nres2 = nxy.reduce(np.add)

nweights = np.dot(np.linalg.inv(nres), nres2.T)

# Make predictions on test with our train weights
pred = test.map(predict(nweights))

# Because we already filtered by code "en"
pred = pred.filter(lambda r: r[1] == "yahoo")
