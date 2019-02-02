import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
from sklearn.cluster import KMeans
import numpy as np

x = np.array([[1,2],[.5,.8],[5,9],[8,9],[11,11],[2,2]])


#plt.scatter(x[:,0],x[:,1],s = 150,linewidths=5)
#plt.show()

clf = KMeans(n_clusters=2)
clf.fit(x)

centroids = clf.cluster_centers_
labels = clf.labels_
colors = 15*['g.','r.']

for i in range(len(x)):
	plt.plot(x[i][0],x[i][1],colors[labels[i]],markersize = 35)

plt.scatter(centroids[:,0],centroids[:,1],marker = 'x',s = 135)
plt.show()

#x_test,x_train,y_test,y_train = 
