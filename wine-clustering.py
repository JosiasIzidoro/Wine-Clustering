import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

base = pd.read_csv('wine-clustering.csv')

x = (base - base.min()) / (base.max() - base.min()) #Normalização de dados
x = pd.DataFrame(x)
b_x = x.iloc[:,[0,1]].values

pca = PCA()
pca_x = pca.fit_transform(x)
pca_x = pd.DataFrame(pca_x)
pca_x = pca_x.iloc[:, [0, 1]].values

plt.figure(figsize=(10,6))
plt.scatter(x=pca_x[:, 0], y=pca_x[:, 1], color='blue',lw=0.1)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.title('Data')
plt.show()

from sklearn.cluster import KMeans

wcss_pca = []
for i in range(1, 11):
    
    kmeans = KMeans(n_clusters = i, random_state = 0)
    kmeans.fit(pca_x)
    wcss_pca.append(kmeans.inertia_)
    
wcss = []
for i in range(1, 11):
    
    kmeans = KMeans(n_clusters = i, random_state = 0)
    kmeans.fit(b_x)
    wcss.append(kmeans.inertia_)
    
fig, ax = plt.subplots(1, 2)
ax[0].plot(range(1, 11), wcss_pca)
ax[0].set_xlabel('número de cluster')
ax[0].set_ylabel('WCSSPCA')
ax[1].plot(range(1, 11), wcss)
ax[1].set_xlabel('número de cluster')
ax[1].set_ylabel('WCSS')

-----------------------------------------------------------------------------------------
from sklearn.metrics import silhouette_score
silhouettePCA = {}
for i in range(2,10):
    kmeans = KMeans(n_clusters = i, random_state = 0, init = 'k-means++')
    kmeans.fit(pca_x)
    silhouettePCA[i] = silhouette_score(pca_x, kmeans.labels_, metric='euclidean')
plt.subplot(1, 2, 1)
plt.bar(range(len(silhouettePCA)), list(silhouettePCA.values()), align='center',color= 'red',width=0.5)
plt.xticks(range(len(silhouettePCA)), list(silhouettePCA.keys()))
plt.grid()
plt.title('Silhouette Score PCA',fontweight='bold')
plt.xlabel('Number of Clusters')
plt.show()

kmeans = KMeans(n_clusters = 3, random_state = 0, init = 'k-means++')
pred = kmeans.fit_predict(pca_x)

plt.subplot(1, 2, 2)
plt.scatter(pca_x[pred == 0, 0], pca_x[pred == 0, 1], s = 20, c = 'red', label = 'Cluster 1', lw = 0.1)
plt.scatter(pca_x[pred == 1, 0], pca_x[pred == 1, 1], s = 20, c = 'orange', label = 'Cluster 2', lw = 0.1)
plt.scatter(pca_x[pred == 2, 0], pca_x[pred == 2, 1], s = 20, c = 'green', label = 'Cluster 3', lw = 0.1)
#plt.scatter(pca_x[pred == 3, 0], pca_x[pred == 3, 1], s = 20, c = 'purple', label = 'Cluster 4', lw = 0.1)
#plt.scatter(pca_x[pred == 4, 0], pca_x[pred == 4, 18], s = 100, c = 'black', label = 'Cluster 5')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()

print('Accuracy: ',silhouette_score(pca_x, kmeans.labels_))

------------------------------------------------------------------------------

silhouette = {}
for i in range(2,10):
    kmeans = KMeans(n_clusters = i, random_state = 0, init = 'k-means++')
    kmeans.fit(b_x)
    silhouette[i] = silhouette_score(b_x, kmeans.labels_, metric='euclidean')
plt.subplot(1, 2, 1)
plt.bar(range(len(silhouette)), list(silhouette.values()), align='center',color= 'red',width=0.5)
plt.xticks(range(len(silhouette)), list(silhouette.keys()))
plt.grid()
plt.title('Silhouette Score',fontweight='bold')
plt.xlabel('Number of Clusters')
plt.show()

kmeans = KMeans(n_clusters = 3, random_state = 0, init = 'k-means++')
pred = kmeans.fit_predict(b_x)

plt.subplot(1, 2, 2)
plt.scatter(b_x[pred == 0, 0], b_x[pred == 0, 1], s = 20, c = 'red', label = 'Cluster 1', lw = 0.1)
plt.scatter(b_x[pred == 1, 0], b_x[pred == 1, 1], s = 20, c = 'orange', label = 'Cluster 2', lw = 0.1)
plt.scatter(b_x[pred == 2, 0], b_x[pred == 2, 1], s = 20, c = 'green', label = 'Cluster 3', lw = 0.1)
#plt.scatter(x[pred == 3, 0], x[pred == 3, 1], s = 20, c = 'purple', label = 'Cluster 4', lw = 0.1)
#plt.scatter(pca_x[pred == 4, 0], pca_x[pred == 4, 18], s = 100, c = 'black', label = 'Cluster 5')
plt.xlabel('Teor alcoolico')
plt.ylabel('Acidez')
plt.legend()

from sklearn.metrics import silhouette_score

print('Accuracy: ',silhouette_score(b_x, kmeans.labels_))

=========================================================================================
