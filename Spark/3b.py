def xy_scale(r):
    return np.dot(r[2].T, r[3])

xy = rdd.map(xy_scale)

res2 = xy.reduce(np.add)
