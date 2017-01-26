import matplotlib.pyplot as plt
import pandas as pd
import sys
import numpy as np

__saved_context__ = {}
#plt.rcParams['agg.path.chunksize'] = 1000
def saveContext():
    import sys
    __saved_context__.update(sys.modules[__name__].__dict__)

def restoreContext():
    import sys
    names = sys.modules[__name__].__dict__.keys()
    for n in names:
        if n not in __saved_context__:
            del sys.modules[__name__].__dict__[n]

def train_gen(data={}):
    for i in data.keys():
	yield [i,data[i]]

def plot_all(data_points):	
	fields = train.keys()
	#print [list(i) for i in train_gen(train)][:20]
	for i in train_gen(train):
		try:
			#saveContext()
			#train = pd.read_hdf('../kaggle_data/train.h5')
			# float(train[i][1])
			i = list(i)
			i[1] = map(lambda x: 0 if x==float('nan') else x,i[1])[:data_points]
			plt.plot(i[1])
			plt.savefig('plots/'+i[0]+'.png')
			#restoreContext()
			plt.clf()
			plt.close()
		except Exception as e:
			print e
			pass

def cor():
	d = {}
	y = train.y
	for i in train_gen(train):
		j = list(i)
		j[1] = map(lambda x: 0 if x==float('nan') else x,j[1])
		c = np.correlate(y,j[1])
		d[j[0]] = c
		print j[0],c

if __name__ == '__main__':
	train = pd.read_hdf('../kaggle_data/train.h5')

	if sys.argv[-1]=='plot':
		plot_all(1000)
	if sys.argv[-1] == 'cor':
		cor()