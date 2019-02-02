import numpy as np
from sklearn import preprocessing, cross_validation,neighbors
import pandas as pd
df = pd.read_csv('breast-cancer-wisconsin.data')
print(df.tail())
df.replace('?',-99999,inplace=True)
df.drop(['id'],1,inplace=True)
print(df.tail())


x = np.array(df.drop(['class'],1))
y = np.array(df['class'])

accuracies = []

for counter in range(25):
	x_train, x_test, y_train,y_test = cross_validation.train_test_split(x,y,test_size = 0.2)
	clf = neighbors.KNeighborsClassifier(n_jobs=-1) #n_jobs by default is 1 # set it to -1 for max multithreading
	clf.fit(x_train,y_train)

	accuracy = clf.score(x_test,y_test)
	print(accuracy)
	accuracies.append(accuracy)
	#exp_measures = np.array([[4,2,1,1,1,2,3,2,1],[4,2,1,2,2,2,3,2,1]]) #custom value by me,,but something not in the dataset already
	#exp_measures = exp_measures.reshape(len(exp_measures),-1) # this line is to make sure the array size is the same for all the real time examples


	#prediction = clf.predict(exp_measures)
	#print(prediction)
print(sum(accuracies)/len(accuracies))
	# and thats it..end of k nearest neighbours..