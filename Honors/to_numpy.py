import matplotlib.image
import numpy as np
from skimage.io import imread
import glob
import os
import sys
import numpy as np


# Assumes that this file is being run in the same directory as the training
# folder is in
fullpaths = glob.glob(os.path.join(sys.argv[1],"*"))
classes = [os.path.split(p)[1] for p in fullpaths]

# First, count how many images we have
numberofImages = 0
for fileNameDir in os.walk(sys.argv[1]):
    for fileName in fileNameDir[2]:
        # Skip non-picture images
        if fileName[-4:] != ".png":
          continue
        numberofImages += 1

# All the pictures are 31x21, and each row of the resulting numpy array has the
# format of [label, pixel1, pixel2, ..., pixel651]
pics = np.zeros((numberofImages, 31*21 + 2), dtype=float)
labels = np.empty((numberofImages,1), dtype='object')

# i is our index in the label and picture arrays
i = 0
for fileNameDir in os.walk(sys.argv[1]):
    for fileName in fileNameDir[2]:
        if fileName[-4:] != ".png":
          continue
        nameFileImage = "{0}{1}{2}".format(fileNameDir[0], os.sep, fileName) 
        # Read the image in as a 1D array
        splitted = fileName.split("_")
        x = int(splitted[1])
        y = int(splitted[2])
        pics[i][0] = x
        pics[i][1] = y
        pics[i][2:] = matplotlib.image.imread(nameFileImage).flatten()
        # Label is just the folder name
        labels[i] = splitted[0] + ".jpg"
        i+=1

outfile = open('imgs.npy','w')
# Put the labels at the beginning
result = np.concatenate((labels, pics), axis=1)
np.save(outfile, result)
