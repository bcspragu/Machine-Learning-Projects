import numpy as np
import matplotlib.pyplot as plt

centers = np.load("model.npy")

data_dir = "/home/bsprague/Projects/CS589/MovieLens/Data/" 
Items   = np.load(data_dir + "items.npy")
Genres  = np.load(data_dir + "genres.npy")

genresSub = [np.zeros(19),np.zeros(19),np.zeros(19)]
genresTotal = [np.zeros(19),np.zeros(19),np.zeros(19)]

for i, center in enumerate(centers):
    for movie_id, rating in enumerate(center):
        if rating >= 4:
            genresSub[i] = np.add(genresSub[i], Items[movie_id][2])
        genresTotal[i] = np.add(genresTotal[i], Items[movie_id][2])

plt.tick_params(axis='both', which='major', labelsize=10)

for i in range(3):
    plt.figure(i+1)
    plt.xticks(np.arange(19), Genres)
    plt.xlabel("Genre")
    plt.ylabel("Portion of Titles")
    plt.title("Portion of 4+ star rated titles by genre in Cluster " + str(i))
    plt.bar(np.arange(19), np.divide(genresSub[i], genresTotal[i]), align='center')

plt.show()
