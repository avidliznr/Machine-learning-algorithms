import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.cluster import KMeans
style.use('ggplot')
from sklearn import preprocessing,model_selection
import pandas as pd
 #FLAT CLUSTERING	

df = pd.read_excel('titanic.xls')
df.drop(['body','name'],1,inplace=True)
df.convert_objects(convert_numeric=True)
df.fillna(0,inplace=True)
#print(df.head())

def handle_non_numeric_data(df):
	columns = df.columns.values
	for column in columns:
		text_value = {}
		def convert_to_int(val):
			return text_value[val]
		if df[column].dtype != np.int64 and df[column].dtype != np.float64:
			column_contents = df[column].values.tolist()
			unqiue_elements = set(column_contents)
			x = 0
			for unique in unqiue_elements:
				if unique not in text_value:
					text_value[unique] = x
					x+=1

			df[column] = list(map(convert_to_int,df[column]))
	return df
df = handle_non_numeric_data(df)
df.drop(['ticket','boat'],1,inplace=True)
#print(df.head())
x = np.array(df.drop(['survived'],1).astype(float))
x = preprocessing.scale(x)
y = np.array(df['survived'])
clf = KMeans(n_clusters=2)
clf.fit(x)
correct = 0


for i in range(len(x)):
	predict_i = np.array(x[i].astype(float))
	predict_i = predict_i.reshape(-1,len(predict_i))
	prediction = clf.predict(predict_i)
	if prediction[0] == y[i]:
		correct += 1
print(correct/len(x))
