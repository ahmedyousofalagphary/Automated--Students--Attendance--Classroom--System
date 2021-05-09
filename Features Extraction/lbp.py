import csv
from os import path
import cv2
import os
from skimage import feature
import numpy as np
class LocalBinaryPatterns:
	def __init__(self, numPoints, radius):
		# store the number of points and radius
		self.numPoints = numPoints
		self.radius = radius
	def describe(self, image, eps=1e-7):
		# compute the Local Binary Pattern representation
		# of the image, and then use the LBP representation
		# to build the histogram of patterns
		lbp = feature.local_binary_pattern(image, self.numPoints,
			self.radius, method="uniform")
		(hist, _) = np.histogram(lbp.ravel(),
			bins=np.arange(0, self.numPoints + 3),
			range=(0, self.numPoints + 2))
		# normalize the histogram
		hist = hist.astype("float")
		hist /= (hist.sum() + eps)
		# return the histogram of Local Binary Patterns
		return hist

desc = LocalBinaryPatterns(100, 8)
data = []
labels = []
# DECLARE PARAMETER
NUM_FEATURES = 100  # this is the number of features to be extracted to a the csv file

# Declare empty container to hold extracted category
category = []

# Declare empty container to hold extracted features
hogArray = []

# Loop through the dataset folder to fetch category folder
label=-1
for folder in os.listdir("Fingerprintdata"):
    label=label+1
    if(folder == ".DS_Store"):
        continue

    # Loop through each category
    for filename in os.listdir(path.join("Fingerprintdata", folder)):


        # Select images which are png and jpg only
        if (filename[-3:] == "png" or filename[-3:] == "jpg" or filename[-3:] == "JPG" or filename[-4:] == "jpeg"):

            # Get full image by joining
            # all the path to the image
            image = path.join("Fingerprintdata", folder, filename)

            # Use open cv to read the image
            img = cv2.imread(image)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Resize the image to (64, 128)
            # Default for hog

            resized = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
            hist = desc.describe(resized)

            category.append(label)

            # append the extracted features of
            # the image to a category container
            hogArray.append(hist)


# convert the extracted features
# from array to vector

# Create a container to hold data to be saved into csv
import pandas as pd
data = np.array(hogArray)
label=np.array(category)
import csv
import cv2
from sklearn.decomposition import PCA
import os
from os import path
import numpy as np
# setup PCA for dimensionality reduction
pca = PCA(n_components=20)
reduced_features = pca.fit_transform(data)
features = reduced_features.tolist()

# Create a container to hold data to be saved into csv
csvData = []
for id, line in enumerate(features):
    newImg = line

    # Prepend the category of each image to
    # the begining of the features
    newImg.insert(0, category[id])
    csvData.append(newImg)


# Save the csv file
with open('Fingerprintdata.csv', 'w') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerows(csvData)
csvFile.close()

print("Done Extracting Features")

