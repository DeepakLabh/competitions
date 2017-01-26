import matplotlib.pyplot as plt
import pandas as pd
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
	
train = pd.read_hdf('../kaggle_data/train.h5')
fields = train.keys()
#print [list(i) for i in train_gen(train)][:20]
for i in train_gen(train):
	try:
		#saveContext()
		#train = pd.read_hdf('../kaggle_data/train.h5')
		# float(train[i][1])
		i = list(i)
		i[1] = map(lambda x: 0 if x==float('nan') else x,i[1])[:10000]
		plt.plot(i[1])
		plt.savefig('plots/'+i[0]+'.png')
		#restoreContext()
		plt.clf()
		plt.close()
	except Exception as e:
		print e
		pass
