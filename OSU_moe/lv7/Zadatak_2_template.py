import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as Image
from sklearn.cluster import KMeans

# ucitaj sliku
img = Image.imread("imgs\\test_1.jpg")

# prikazi originalnu sliku
plt.figure()
plt.title("Originalna slika")
plt.imshow(img)
plt.tight_layout()
plt.show()

# pretvori vrijednosti elemenata slike u raspon 0 do 1
img = img.astype(np.float64) / 255

# transfromiraj sliku u 2D numpy polje (jedan red su RGB komponente elementa slike)
w,h,d = img.shape
img_array = np.reshape(img, (w*h, d))

# rezultatna slika
img_array_aprox = img_array.copy()



km = KMeans(n_clusters=5, init="k-means++", n_init=5, random_state=0)
km.fit(img_array_aprox)
labels = km.predict(img_array_aprox)

centroids = km.cluster_centers_

img_array_aprox[:, 0] = centroids[labels][:, 0]
img_array_aprox[:, 1] = centroids[labels][:, 1]
img_array_aprox[:, 2] = centroids[labels][:, 2]
img_array_aprox = np.reshape(img_array_aprox, (w, h, d))

f, axarr = plt.subplots(1, 2)
axarr[0].imshow(img)
axarr[1].imshow(img_array_aprox)
plt.tight_layout()
plt.show()

"""km = KMeans(n_clusters=5, init="k-means++", n_init=5, random_state=0):
This initializes a KMeans clustering model with 5 clusters.
init="k-means++" specifies that the initial cluster centers should be selected using the k-means++ algorithm, which improves the convergence of the algorithm.
n_init=5 specifies the number of times the KMeans algorithm will be run with different centroid seeds. 
The final results will be the best output of n_init consecutive runs in terms of inertia (sum of squared distances to the nearest centroid).
random_state=0 sets the random seed for reproducibility.

km.fit(img_array_aprox):
This fits the KMeans model to the data img_array_aprox, where each row represents a data point (in this case, pixels of an image).

labels = km.predict(img_array_aprox):
This predicts the cluster labels for each data point based on the trained KMeans model.

centroids = km.cluster_centers_:
This retrieves the coordinates of the cluster centroids.

img_array_aprox[:, 0] = centroids[labels][:, 0], img_array_aprox[:, 1] = centroids[labels][:, 1], img_array_aprox[:, 2] = centroids[labels][:, 2]:
These lines assign the RGB values of the cluster centroids to each pixel in the img_array_aprox array based on the predicted cluster labels.
img_array_aprox = np.reshape(img_array_aprox, (w, h, d)):
This reshapes the modified img_array_aprox back into the original image dimensions (w, h, d).

f, axarr = plt.subplots(1, 2):
This creates a figure and a set of subplots. It specifies that there will be 1 row and 2 columns of subplots.

axarr[0].imshow(img):
This displays the original image in the first subplot.
axarr[1].imshow(img_array_aprox):
This displays the modified image (after clustering and recoloring) in the second subplot.

plt.tight_layout():
This adjusts the layout of the subplots to prevent overlapping.

plt.show():
This displays the figure with the original and modified images."""
