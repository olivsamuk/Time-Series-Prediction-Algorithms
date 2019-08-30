import tensorflow as tf
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.contrib.rnn import BasicRNNCell
import time
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
import sys
import json
import itertools

def write_list(lista, path, title):
    with open(path, 'w') as f:
        f.write(str(title)+"=[")
        for item in lista:
            f.write("%s" % item)
            f.write(", ")
        f.write("]")

path = sys.argv[1]
beginTime = time.time()

# ------ Call data from CSV File
# txt_data = pd.read_csv(path)
# # txt_data.sort_values(["date", "hour"], inplace=True)
# txt_data.head()
# x = txt_data[['target']]

# data_lenght = len(txt_data)
# print("data_lenght: ", data_lenght)

# ------ Call data from LIST File
f = open(sys.argv[1], "r")
lst_temp = f.read()
lst = lst_temp.split(',')
del lst[-1]
x = [float(i) for i in lst]

data_lenght = len(txt_data)
print("data_lenght: ", data_lenght)
# -------------------------------

n_steps = 20

for time_step in range(1, n_steps+1):
    txt_data['target'+str(time_step)] = txt_data[['target']].shift(-time_step).values

txt_data.dropna(inplace=True)
txt_data.head()

# X
txt_data.iloc[:, :n_steps]
# y
txt_data.iloc[:, 1:]

X = txt_data.iloc[:, :n_steps].values
X = np.reshape(X, (X.shape[0], n_steps, 1)) # add dimensão

y = txt_data.iloc[:, 1:].values
y = np.reshape(y, (y.shape[0], n_steps, 1)) # add dimensão

txt_data.shape

n_test = int(data_lenght * .75)

X_train, X_test = X[:-n_test, :, :], X[-n_test:, :, :]
y_train, y_test = y[:-n_test, :, :], y[-n_test:, :, :]

shuffle_mask = np.arange(0, X_train.shape[0])
np.random.shuffle(shuffle_mask)

X_train = X_train[shuffle_mask]
y_train = y_train[shuffle_mask]

n_inputs = 1
n_neurons = 64
n_outputs = 1
learning_rate = 0.001

graph = tf.Graph()
with graph.as_default():
    # placeholders
    tf_X = tf.placeholder(tf.float32, [None, n_steps, n_inputs], name='X')
    tf_y = tf.placeholder(tf.float32, [None, n_steps, n_outputs], name='y')
    
    with tf.name_scope('Recurent_Layer'):
        cell = BasicRNNCell(num_units=n_neurons, activation=tf.nn.elu)
        outputs, Last_state = tf.nn.dynamic_rnn(cell, tf_X, dtype=tf.float32)
             
    with tf.name_scope('out_layer'):
        stacked_outputs = tf.reshape(outputs, [-1, n_neurons])
        stacked_outputs = tf.layers.dense(stacked_outputs, n_outputs, activation=None)
        net_outputs = tf.reshape(stacked_outputs, [-1, n_steps, n_outputs])
    
    with tf.name_scope('train'):
        loss = tf.reduce_mean(tf.abs(net_outputs - tf_y)) # MAE
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    init = tf.global_variables_initializer()

for x in range(0,10):

    batch_size = 64
    iterations_list, mae_list = [], [] # For graphic plots
    LastMaeChecked = 0
    SameMaeIterationsCounter = 0

    with tf.Session(graph=graph) as sess:
        init.run()
        
        for step in itertools.count():
            
            offset = (step * batch_size) % (y_train.shape[0] - batch_size)
            X_batch = X_train[offset:(offset + batch_size), :]
            y_batch = y_train[offset:(offset + batch_size)]
            
            sess.run(optimizer, feed_dict={tf_X: X_batch, tf_y: y_batch})

            iterations_list.append(step)        
            mae_list.append(loss.eval(feed_dict={tf_X: X_train, tf_y: y_train}))
            
            if (loss.eval(feed_dict={tf_X: X_train, tf_y: y_train}) < LastMaeChecked) or (step == 0):
                LastMaeChecked = loss.eval(feed_dict={tf_X: X_train, tf_y: y_train})
                SameMaeIterationsCounter = 0
            else:
                SameMaeIterationsCounter+=1

            if SameMaeIterationsCounter == 1000 or step == 10000:
                break
     
            if step % 2000 == 0:
                train_mae = loss.eval(feed_dict={tf_X: X_train, tf_y: y_train})
                print(step, "\tTrain MAE:", train_mae)

        test_mae = loss.eval(feed_dict={tf_X: X_test, tf_y: y_test})

        y_pred = sess.run(net_outputs, feed_dict={tf_X: X_test})

    endTime = time.time() - beginTime
    r2 = r2_score(y_pred=y_pred.reshape(-1,1), y_true=y_test.reshape(-1,1))

    results = {'dataset':str(path), 'time':endTime, 'mae':mean_absolute_error(y_pred[:n_test,0,0],y_test[:n_test,0,0]), 'r2':r2, 'iteracoes': len(iterations_list)}
    print(results)

    #Write Results in Json ---------------------------------
    data = json.load(open('resultsRNN.json'))
    if type(data) is dict:
        data = [data]
    data.append(results)
    with open('resultsRNN.json', 'w') as f:
      json.dump(data, f)

# write_list(y_pred[:n_test,0,0], "rnn_orig/"+path[8:-7]+"pred.txt", "predictions")
# write_list(y_test[:n_test,0,0], "rnn_orig/"+path[8:-7]+"test.txt", "test")

sample = 10
n = 200 

plt.style.use("ggplot")
f = plt.figure(figsize=(10,7))
plt.plot(range(n), y_pred[:n,sample,0], label="Real")
plt.plot(range(n), y_test[:n,sample,0], color="yellow", label="Prevista")
plt.ylabel("txt_data (qtd)")
plt.xlabel("Tempo")
plt.legend(loc="best")
f.savefig("rnn_txt_data1.png")
plt.show()        
