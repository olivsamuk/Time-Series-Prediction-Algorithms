# -*- coding: utf-8 -*-

from pandas import read_csv
from pandas import datetime
from pandas import DataFrame
from statsmodels.tsa.arima_model import ARIMA
from matplotlib import pyplot
from pandas.tools.plotting import autocorrelation_plot
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np
import time
import sys
import json

def write_list(lista, path, title):
    with open(path, 'w') as f:
        f.write(str(title)+"=[")
        for item in lista:
            f.write("%s" % item)
            f.write(", ")
        f.write("]")

path = sys.argv[1]
beginTime = time.time()

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
train, test = X[0:n_test], X[n_test:len(X)]

history = [x for x in train]
predictions = list()
for t in range(len(test)):
	model = ARIMA(history, order=(0,1,0))
	model_fit = model.fit()
	output = model_fit.forecast()
	yhat = output[0]
	predictions.append(yhat)
	obs = test[t]
	history.append(obs)
	print('predicted=%f, expected=%f' % (yhat, obs))

error = mean_absolute_error(test, predictions)
end = time.time() - beginTime
r2 = r2_score(test, predictions)

results = {'dataset':str(path), 'mae':round(error, 3), 'r2':round(r2, 3), 'time':end}
print(results)

#Write Results in Json ---------------------------------
data = json.load(open('resultsARIMA.json'))
if type(data) is dict:
    data = [data]
data.append(results)
with open('resultsARIMA.json', 'w') as f:
  json.dump(data, f)


font = {'family' : 'Times New Roman'}
pyplot.rc('font', **font)

pyplot.rc('font', size=24)          # controls default text sizes
pyplot.rc('axes', titlesize=24)     # fontsize of the axes title
pyplot.rc('axes', labelsize=24)    # fontsize of the x and y labels
pyplot.rc('xtick', labelsize=24)    # fontsize of the tick labels
pyplot.rc('ytick', labelsize=24)    # fontsize of the tick labels
pyplot.rc('legend', fontsize=24)    # legend fontsize
pyplot.rc('figure', titlesize=24)  # fontsize of the figure title

pyplot.style.use("ggplot")


f = pyplot.figure()

pyplot.plot(test, color='red', label="Real", linestyle=":")
pyplot.plot(predictions, color='yellow', label="Prevista")
pyplot.ylabel("Variável", fontsize=32, fontweight='bold')
pyplot.xlabel("Tempo (minutos)", fontsize=32, fontweight='bold')
pyplot.legend(loc="best")
# f.savefig(sys.argv[1]+"_ARIMA.png")
pyplot.show()