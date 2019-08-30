# -*- coding: utf-8 -*-
"""
Created on Thu May 18 00:58:08 2017

@author: biank
"""

from pandas import read_csv
from pandas import datetime
from pandas import DataFrame
from statsmodels.tsa.arima_model import ARIMA
from matplotlib import pyplot as plt
from pandas.tools.plotting import autocorrelation_plot
from sklearn.metrics import mean_absolute_error
import numpy as np
import time
import sys
from sklearn.metrics import r2_score
from sklearn.svm import SVR
import json
import pandas as pd

def write_list(lista, path, title):
    with open(path, 'w') as f:
        f.write(str(title)+"=[")
        for item in lista:
            f.write("%s" % item)
            f.write(", ")
        f.write("]")

path = sys.argv[1]
begin = time.time()

# ------ Call data from CSV File
# bike_data = pd.read_csv(path)
# # bike_data.sort_values(["dteday", "hr"], inplace=True)
# bike_data.head()
# X = bike_data[['target']].values

# data_lenght = len(X)
# print("data_lenght: ", data_lenght)

# # ------ Call data from LIST File
f = open(sys.argv[1], "r")
lst_temp = f.read()
lst = lst_temp.split(',')
del lst[-1]
X = [float(i) for i in lst]

data_lenght = len(X)
print("data_lenght: ", data_lenght)
# # -------------------------------
 
n_test = int(data_lenght * .75)
train, test = X[(n_test-100):n_test], X[n_test:len(X)]


history = [x for x in train]
predictions = list()
for t in range(len(test)):
    test_data = np.arange(len(history),len(history)+1)
    test_data = np.expand_dims(test_data,axis=1)
    train_data = np.arange(0,len(history))
    train_data = np.expand_dims(train_data,axis=1)
    svr = SVR(kernel='rbf', C=1e3, gamma = 1/1250)
    yhat = svr.fit(train_data,history).predict(test_data)
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)
    del history[0]
    print('predicted=%f, expected=%f' % (yhat, obs))

error = mean_absolute_error(test, predictions)
end = time.time() - begin
r2 = r2_score(test, predictions)


results = {'dataset':str(path), 'mae':round(error, 3), 'r2':round(r2, 3), 'time':end}
print(results)

#Write Results in Json ---------------------------------
data = json.load(open('resultsSVM.json'))
if type(data) is dict:
    data = [data]
data.append(results)
with open('resultsSVM.json', 'w') as f:
  json.dump(data, f)


# write_list(predictions, "SVM_rec/"+path[8:-4]+"_pred.txt", "predictions")
# write_list(test, "SVM_rec/"+path[8:-4]+"_test.txt", "test")



font = {'family' : 'Times New Roman'}
plt.rc('font', **font)

plt.rc('font', size=24)          # controls default text sizes
plt.rc('axes', titlesize=24)     # fontsize of the axes title
plt.rc('axes', labelsize=24)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=24)    # fontsize of the tick labels
plt.rc('ytick', labelsize=24)    # fontsize of the tick labels
plt.rc('legend', fontsize=24)    # legend fontsize
plt.rc('figure', titlesize=24)  # fontsize of the figure title

plt.style.use("ggplot")
f = plt.figure(figsize=(10,7))
plt.plot(test, color='red', label="Real", linestyle=":")
plt.plot(predictions, color='yellow', label="Prevista")
plt.ylabel("Variável", fontsize=32, fontweight='bold')
plt.xlabel("Tempo (minutos)", fontsize=32, fontweight='bold')
plt.legend(loc="best")
# f.savefig(sys.argv[1]+"_SVM.png")
plt.show()

