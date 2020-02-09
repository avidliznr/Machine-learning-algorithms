import json
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import pandas as pd
import math
import numpy as np
from sklearn import preprocessing,svm
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, BayesianRidge

with open('60DayStopRandomCountV2.json', 'r') as f:
    passengerData = json.load(f)

data_dict = dict()
stop_count = list()

for day in passengerData:
    stops = passengerData.get(day)
    stopCountDict = dict()
    for stop in stops:
        total_val = 0
        timeVals = stops.get(stop)
        for time in timeVals:
            total_val += timeVals.get(time)
        stopCountDict.update({stop: total_val})
    data_dict.update({day: stopCountDict})

for day in data_dict:
    stop_count.append(data_dict.get(day).get('Thiyagaraya Nagar Bus Depot'))


y = np.asarray(stop_count)
x = np.asarray(range(len(stop_count)))
print(len(x),len(y))
x = x.reshape(-1, 1)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=.2)
#print(len(x_train),len(y_train))
#print(x_train,y_train)
reg = LinearRegression()
reg.fit(x_train,y_train)
print('fitting done')

#loaded = pickle.load(open())

predicted = []
predicted = np.ndarray.tolist(y)
for abc in range(60,90):
    predicted.append(reg.predict([[abc]]))




accuracy = reg.score(x_test,y_test)
print(accuracy)
#normal data plot
plt.subplot(2,1,1)
plt.plot(x,y)
plt.ylim(0,2000)
plt.xlim(0,100)
plt.xlabel('days')
plt.ylabel('total passengers')
plt.title('original data')
#predicted data plot
new_x = range(len(predicted))
plt.subplot(2,1,2)
plt.plot(new_x,predicted)
plt.ylim(0,2000)
plt.xlim(0,100)
plt.xlabel('days')
plt.ylabel('total passengers')
plt.title('predicted data 1')

plt.show()

print('process ended')
