def x_xtranspose(r):
    two_dim = r[2]
    return np.dot(two_dim, two_dim.T)

xxt = rdd.map(x_xtranspose)

# res holds the 18x18 matrix

res = xxt.reduce(np.add)
