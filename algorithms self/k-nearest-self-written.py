# self written code for k neaerest neighbours by calculation the euclidian distance
from math import *
import numpy as np
import matplotlib.pyplot as plt
import warnings
from matplotlib import style
style.use('fivethirtyeight')
from collections import Counter
import random
import pandas as pd
#print(df.head())
#data created by me for checking purpouses
#data = {'k':[[1,2],[2,3],[3,1]],'r':[[6,5],[7,7],[8,6]]}
#new_feautures = [5,7]

#for i in dataser:
#	for ii in data[i]:
#		plt.scatter(ii[0],ii[1],s=100,color = i)

#the same above for loop using list comprehensions

'''[[plt.scatter(ii[0],ii[1],s=100,color=i) for ii in data[i]] for i in data]

plt.scatter(new_feautures[0],new_feautures[1],s = 100) #s -> size
plt.show()'''


def k_nearest_neighbors(data,predict,k=3):
	if len(data) >= k:
		warnings.warn('enter a good data mathafaka\nk value is set to a value less than total voting groups')
	distances=[]
	for group in data:
		for feautures in data[group]:
			ed = np.linalg.norm(np.array(feautures)- np.array(predict))
			distances.append([ed,group])

	votes = [i[1] for i in sorted(distances)[:k]]
	#print(Counter(votes).most_common(1))
	vote_results = Counter(votes).most_common(1)[0][0]
	confidence = Counter(votes).most_common(1)[0][1]/k

	return vote_results,confidence

accuracies = []
for i in range(25):
	#using the uci data on breast cancer
	df = pd.read_csv('breast-cancer-wisconsin.data')
	df.replace('?',-99999,inplace=True)
	df.drop(['id'],1,inplace = True)
	full_data = df.astype(float).values.tolist()
	#print(full_data[:10])
	random.shuffle(full_data)

	#checking if data is shuffled
	#print(20*'#')
	#print(full_data[:10])

	test_size = 0.4
	train_set = {2:[],4:[]}
	test_set = {2:[],4:[]}
	train_data = full_data[:-int(test_size*len(full_data))]
	test_data = full_data[-int(test_size*len(full_data)):]

	for i in train_data:
		train_set[i[-1]].append(i[:-1])

	for i in test_data:
		test_set[i[-1]].append(i[:-1])

	correct,total = 0,0

	for group in test_set:
		for data in test_set[group]:
			votes,confidence = k_nearest_neighbors(train_set,data,k=5)
			if group == votes:
				correct +=1
			total +=1
	accuracies.append(correct/total)

	print('accuracy is :',correct/total)
print(sum(accuracies)/len(accuracies))



	