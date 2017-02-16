import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
from sklearn import ensemble, linear_model

train = pd.read_hdf('../kaggle_data/train.h5')

ids = train.id.unique()
ids_sorted = sorted(ids)
least_id = ids_sorted[-1]
model_dict = {}

train_mean = train.mean

cols_train = ['technical_20', 'technical_30', ]