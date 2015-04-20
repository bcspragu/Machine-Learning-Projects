resMovies = [{},{},{}]
top1 = [[],[],[]]
top2 = [[],[],[]]
for i, x in enumerate(clusters):
    ratings = zip(Xtrain[i].indices, Xtrain[i].data)
    for r in ratings:
        if r[0] in resMovies[int(x)]:
            resMovies[int(x)][r[0]].append(r[1])
        else:
            resMovies[int(x)][r[0]] = [r[1]]

for i, cl in enumerate(resMovies):
    for movie_id, ratings in cl.iteritems():
        resMovies[i][movie_id] = [np.average(ratings), len(ratings), Items[movie_id][1]]
    top1[i] = list(sorted(resMovies[i].values()))[:20]
    top2[i] = filter(lambda x: x[1] > 3, list(sorted(resMovies[i].values())))[:20]
