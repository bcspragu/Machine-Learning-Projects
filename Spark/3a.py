def x_xtranspose(r):
    two_dim = r[2].reshape(1,18)
    return np.dot(two_dim.T, two_dim)

xxt = rdd.map(x_xtranspose)

# res holds the 18x18 matrix

res = xxt.reduce(np.add)
