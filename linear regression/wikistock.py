import pandas as pd
import quandl as quandl
import math
import numpy as np
from sklearn import preprocessing,cross_validation,svm
from sklearn.linear_model import LinearRegression
import datetime
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import pickle


quandl.ApiConfig.api_key = 'vygKPQxSLx4-PxACcdht'

df = quandl.get('WIKI/GOOGL')
df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]
df['hl_percent'] = (df['Adj. High'] - df['Adj. Low'])/ df['Adj. Low'] *100.0
df['percent_change'] = (df['Adj. Close'] - df['Adj. Open'])/ df['Adj. Open'] *100.0

df = df[['Adj. Close','hl_percent','percent_change','Adj. Volume']]

forecast_col = 'Adj. Close'
df.fillna(-99999, inplace=True)

forecast_out = int(math.ceil(0.01*len(df))) #0.1 can be changed

df['label'] = df[forecast_col].shift(-forecast_out)
#print(df.head())
#print(df.tail())

# x -> features y-> labels

x = np.array(df.drop(['label'],1))
x = preprocessing.scale(x)
x_lately = x[-forecast_out:]
x = x[:-forecast_out]

df.dropna(inplace=True)
y = np.array(df['label'])

#print(len(x),len(y))
x_train,x_test,y_train,y_test = cross_validation.train_test_split(x,y,test_size=.2)

#change the below line to change the type of model used
#clf = LinearRegression()

clf = LinearRegression(n_jobs=-1)
#n_jobs defines how many jobs can be threaded at the same time..it differs form model to model..some may not even be treadable
#-1 runs as many as possible
#we are using the linear regression classifier 
# this line of code uses the svm regresssion classifers
#clf = svm.SVR() #default kernel is linear we can set it to different values
#clf = svm.SVR(kernel = 'poly')

clf.fit(x_train,y_train)
#after training the dataset it is being saved so that we need not retrain the model every time.now after training it once we can use the
#pickeled form to make the prediction rather than retraining the model
with open('linearregression.pickle','wb') as f:
	pickle.dump(clf,f)

pickle_in = open('linearregression.pickle','rb')
clf = pickle.load(pickle_in)

accuracy = clf.score(x_test,y_test)
#accuracy in 0.97... for linear regression
#for svm.SVR it is 0.807...
#for polynomial kernel it is 0.062..so fuckin worse..

#print(accuracy) # shifted one percent of the data we get an accuracy of this value|| one percent came from the .01*.... .01*100 = 1
#accuracy is actually error squared

forecast_set = clf.predict(x_lately)

print(forecast_set,accuracy,forecast_out)

df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
	next_date = datetime.datetime.fromtimestamp(next_unix)
	next_unix += one_day
	df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]

print(df.tail())

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
