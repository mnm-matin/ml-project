
# Import required libraries
import numpy as np
import pandas as pd
import cv2
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.pairwise import cosine_similarity

# Load images
img1 = cv2.imread('image1.jpg')
img2 = cv2.imread('image2.jpg')
img3 = cv2.imread('image3.jpg')

# Convert images to grayscale
gray_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
gray_img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)

# Reshape images to 1D array
flat_img1 = gray_img1.flatten()
flat_img2 = gray_img2.flatten()
flat_img3 = gray_img3.flatten()

# Concatenate image arrays into a single array
concatenated_images = np.concatenate((flat_img1, flat_img2, flat_img3))

# Apply MiniBatchKMeans to find clusters
clusters = MiniBatchKMeans(n_clusters=3).fit(concatenated_images.reshape(-1, 1))

# Calculate cosine similarity between each image and cluster centroids
centroid1 = clusters.cluster_centers_[0].reshape(1, -1)
centroid2 = clusters.cluster_centers_[1].reshape(1, -1)
centroid3 = clusters.cluster_centers_[2].reshape(1, -1)

similarity1 = cosine_similarity(gray_img1.reshape(1, -1), centroid1)
similarity2 = cosine_similarity(gray_img1.reshape(1, -1), centroid2)
similarity3 = cosine_similarity(gray_img1.reshape(1, -1), centroid3)

# Return most similar image
if similarity1 > similarity2 and similarity1 > similarity3:
    print('Image 1 is most similar to given image')
elif similarity2 > similarity1 and similarity2 > similarity3:
    print('Image 2 is most similar to given image')
else:
    print('Image 3 is most similar to given image')
