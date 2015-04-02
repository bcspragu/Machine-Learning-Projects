base = "/wikistates/{0}"
rdd = sc.union([sc.textFile(base.format(x)) for x in range(6,24)])

