from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('fivethirtyeight')
import random


#xs = np.array([1,2,3,4,5,6,], dtype=np.float64)
#ys = np.array([5,4,6,5,6,7], dtype=np.float64)

def best_fit_slope_and_b(xs,ys):
	m = ( ((mean(xs)*mean(ys)) - mean(xs*ys) ) /
		(mean(xs)*mean(xs) - mean(xs*xs) ) )
	b = mean(ys) - m*mean(xs)
	return m,b


def create_dataset(hm,variance,step=2,corellation=False):
	val = 1
	ys =[]
	xs=[]
	for i in range(hm):
		y = val + random.randrange(-variance,variance)
		ys.append(y)
		if corellation and corellation == 'pos':
			val+=step
		elif corellation and corellation == 'neg':
			val -= step
		xs = [i for i in range(len(ys))]
	return np.array(xs,dtype=np.float64),np.array(ys, dtype=np.float64)

#print(m,b)
def squared_error(ys_original,ys_line):
	return sum((ys_line-ys_original)**2)

def coefficient_of_determination(ys_original,ys_line):
	y_mean_line = [mean(ys_original) for y in ys_original]
	squared_error_regression_line = squared_error(ys_original,ys_line)
	squared_error_y_mean = squared_error(ys_original,y_mean_line)
	return 1 - (squared_error_regression_line / squared_error_y_mean)


xs,ys = create_dataset(40,10,2,corellation='pos')

m,b = best_fit_slope_and_b(xs,ys)

#predictin a sample value
predict_x = 8
predict_y = m*predict_x +b
#end of sample value prediction


regression_line = [(m*x) + b for x in xs]
#print(regression_line)

r_squared = coefficient_of_determination(ys,regression_line)
print(r_squared)


plt.plot(regression_line)
plt.scatter(predict_x,predict_y,color = 'g')
plt.scatter(xs,ys)
plt.show()
