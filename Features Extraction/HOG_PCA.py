import csv
import cv2
from sklearn.decomposition import PCA
import os
from os import path
import numpy as np
hog = cv2.HOGDescriptor()
# DECLARE PARAMETER
NUM_FEATURES = 10  # this is the number of features to be extracted to a the csv file

# Declare empty container to hold extracted category
category = []

# Declare empty container to hold extracted features
hogArray = []

# Loop through the dataset folder to fetch category folder
for folder in os.listdir("facerecognitiondata"):

    if(folder == ".DS_Store"):
        continue

    # Loop through each category
    for filename in os.listdir(path.join("facerecognitiondata", folder)):

        # Select images which are png and jpg only
        if (filename[-3:] == "png" or filename[-3:] == "jpg" or filename[-3:] == "JPG" or filename[-4:] == "jpeg"):

            # Get full image by joining
            # all the path to the image
            image = path.join("facerecognitiondata", folder, filename)

            # Use open cv to read the image
            img = cv2.imread(image)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            resized = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)

            # Compute the Hog Features
            h = hog.compute(resized)

            # Transpose the result from a vector to an array
            hogImage = h.T

            # append the category of the
            # image to a category container
            category.append(folder)

            # append the extracted features of
            # the image to a category container
            hogArray.append(hogImage)


# convert the extracted features
# from array to vector
hogArray_np = np.array(hogArray)

# Reshaped the Features to the acurrate size
reshaped_hog_Array = np.reshape(
    hogArray_np, (hogArray_np.shape[0], hogArray_np.shape[2]))


# setup PCA for dimensionality reduction
pca = PCA(n_components=NUM_FEATURES)
reduced_features = pca.fit_transform(reshaped_hog_Array)
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
with open('facerecognitiondata.csv', 'w') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerows(csvData)
csvFile.close()

print("Done Extracting Features")
