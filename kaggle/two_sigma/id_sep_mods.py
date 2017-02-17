import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
from sklearn import ensemble, linear_model
from collections import Counter
train = pd.read_hdf('../kaggle_data/train.h5')

id_all = train.id
id_count = Counter(id_all)
ids_similar = map(lambda y: y[0], filter(lambda x:x[1]==1813, id_count.most_common(10000)))
ids_similar_value = 1813 ## no of rows in all similar retrieved values.

ids_un = train.id.unique()
ids_sorted = sorted(ids_un)
least_id = ids_sorted[-1]
model_dict = {}

train_mean = train.mean

# cols_train = ['technical_20', 'technical_30', ]
class Features:

	def __init__(self, ids_similar):
		self.ids_similar = ids_similar

	def fit(self, x, y):
		model = linear_model.LinearRegression()
		model.fit(x, y)
		self.model = model

	def predict(self, x):
		return self.model.predict(x)

a = Features(ids_similar)
print(a)