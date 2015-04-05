def xy_scale(r):
    return np.dot(r[2].reshape(1,18), r[3])

xy = rdd.map(xy_scale)

res2 = xy.reduce(np.add)
