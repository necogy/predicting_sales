# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 16:25:14 2018

@author: user
"""
#########################loading libraries###################################
import os
from time import time
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from keras.layers import Dense
import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import TensorBoard
from pandas_summary import DataFrameSummary
from keras.models import Sequential
from keras.layers import Dense

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.metrics import mean_squared_error
from keras import backend as K
from sklearn import preprocessing
from keras.wrappers.scikit_learn import KerasRegressor

os.chdir('C:\\Users\\user\\Documents\\Xuan\\forecasting')
sales = pickle.load(open( "C:\\Users\\user\\Documents\\Xuan\\forecasting\\sku_weekly_store_sales_nikkei_added.p", "rb" ))

last = pd.read_pickle('prepared_full_nn.p')

static = last.drop(['part_no', 'department'], axis=1)

# Normalization: another option:preprocessing.scale
static.loc[:,'cost':'average_high_temp'] =preprocessing.normalize(static.loc[:,'cost':'average_high_temp'])
# a little bit more adjusting: is_
static.loc[:,'is_plain':'is_silky'] = static.loc[:,'is_plain':'is_silky'].astype(int)
static.loc[:,'is_plain':'is_silky'] = static.loc[:,'is_plain':'is_silky'].replace(0,-1)
static.loc[:,'sex'] = static.loc[:,'sex'] .astype(int)
static.loc[:,'sex'] = static.loc[:,'sex'].replace(0,-1)

seed = 7
np.random.seed(seed)

time_varying_part=sales.loc[:,'last_week_sales':'change_in_customers'] 

static.reset_index(drop=True, inplace=True)
time_varying_part.reset_index(drop=True, inplace=True)
#complete = pd.concat([rest,time_varying_part,static], axis=1)
complete = pd.concat([time_varying_part,static], axis=1)
complete.shape
complete.to_pickle('complete_rnn.p')

X = complete.iloc[:,0:complete.shape[1]-1]
Y = complete.iloc[:,complete.shape[1]-1]

x_dim = X.shape[1]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=seed)

scale = StandardScaler()
scaled_X_train=scale.fit_transform(X_train)
scaled_X_test=scale.fit_transform(X_test)

def base_model():
     model = Sequential()
     model.add(Dense(x_dim, input_dim=x_dim, kernel_initializer='normal', activation='relu'))
     model.add(Dense(50, kernel_initializer='normal', activation='relu'))
     model.add(Dense(5, kernel_initializer='normal', activation='relu'))
     model.add(Dense(1, kernel_initializer='normal', activation='linear', name='output_layer'))
     model.compile(loss='mean_squared_error', optimizer = 'adam')
     return model

clf = KerasRegressor(build_fn=base_model, nb_epoch=300, batch_size=10,verbose=1)

clf.fit(scaled_X_test,y_test)

pred = clf.predict(scaled_X_test)
pred = np.round(pred).astype(int)
from sklearn.metrics import accuracy_score
accuracy_score(y_test, pred)

########################saving file####################################################
#get the week & part_no
type(sales.iloc[y_test.index][['week','part_no']])

pd.concat([y_test, pd.Series(pred)], axis=1).reset_index().shape
for_combination=y_test.reset_index(drop=True, inplace=True)

array = pd.concat([pd.Series(y_test.index,name = 'Original_Index'),
                   sales.iloc[y_test.index][['week','part_no']].reset_index(),
                   y_test.reset_index(), 
                   pd.Series(pred,name='prediction')], axis=1)

(y_test.reset_index(drop=True, inplace=True) + pd.Series(pred)).shape

array.to_csv("C:\\Users\\user\\Documents\\Xuan\\forecasting\\y_test_y_hat_comparison", sep='\t', encoding='utf-8')
writer = pd.ExcelWriter('y_test_y_hat_comparison.xlsx')
array.to_excel(writer,'Sheet1')
writer.save()