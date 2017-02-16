import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
from sklearn import ensemble, linear_model
from collections import Counter
train = pd.read_hdf('../kaggle_data/train.h5')

id_all = train.id
id_count = Counter(id_all)
ids_similar = filter(lambda x:x[1]==1813, id_count.most_common(10000))

ids_un = train.id.unique()
ids_sorted = sorted(ids_un)
least_id = ids_sorted[-1]
model_dict = {}

train_mean = train.mean

cols_train = ['technical_20', 'technical_30', ]