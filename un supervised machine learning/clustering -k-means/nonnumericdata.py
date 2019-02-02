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
print(df.head())
