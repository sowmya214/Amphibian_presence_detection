#set up
import csv
import math
import random
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn import preprocessing
logging.basicConfig(level=logging.DEBUG)

#extracting data
frogs_train = pd.read_csv("/content/gdrive/My Drive/dataset.csv", sep=';', header=None, names=['ID', 'Motorway', 'SR', 'NR', 'TR', 'VR', 'SUR1', 'SUR2', 'SUR3', 'UR', 'FR', 'OR', 'RR', 'BR', 'MR', 'CR', 'Green frogs', 'Brown frogs', 'Common toad', 'Fire-bellied toad', 'Tree frog', 'Common newt', 'Great crested newt' ])
copy = frogs_train.copy()

#extracting features (x values)
frog_features = frogs_train.drop(columns=['ID', 'Motorway','Green frogs', 'Brown frogs', 'Common toad', 'Fire-bellied toad', 'Tree frog', 'Common newt', 'Great crested newt' ])
frog_features = frog_features.drop([0,1])

#extracting labels (y, values)
frog_labels = copy.drop(columns=['ID', 'Motorway', 'SR', 'NR', 'TR', 'VR', 'SUR1', 'SUR2', 'SUR3', 'UR', 'FR', 'OR', 'RR', 'BR', 'MR', 'CR'])
frog_labels = frog_labels.drop([0,1])

#preprocessing data (data is being read in as string, converting it to number, then float.)

number = preprocessing.LabelEncoder()

frog_features['SR'] = number.fit_transform(frog_features['SR'])
frog_features['TR'] = number.fit_transform(frog_features['TR'])
frog_features['VR'] = number.fit_transform(frog_features['VR'])
frog_features['SUR1'] = number.fit_transform(frog_features['SUR1'])
frog_features['SUR2'] = number.fit_transform(frog_features['SUR2'])
frog_features['UR'] = number.fit_transform(frog_features['UR'])
frog_features['FR'] = number.fit_transform(frog_features['FR'])
frog_features['OR'] = number.fit_transform(frog_features['OR'])
frog_features['RR'] = number.fit_transform(frog_features['RR'])
frog_features['BR'] = number.fit_transform(frog_features['BR'])

frog_features['SR'] = frog_features['SR'].astype('float32')
frog_features['TR'] = frog_features['TR'].astype('float32')
frog_features['VR'] = frog_features['VR'].astype('float32')
frog_features['SUR1'] = frog_features['SUR1'].astype('float32')
frog_features['SUR2'] = frog_features['SUR2'].astype('float32')
frog_features['UR'] = frog_features['UR'].astype('float32')
frog_features['FR'] = frog_features['FR'].astype('float32')
frog_features['OR'] = frog_features['OR'].astype('float32')
frog_features['RR'] = frog_features['RR'].astype('float32')
frog_features['BR'] = frog_features['BR'].astype('float32')

frog_labels['Brown frogs'] = frog_labels['Brown frogs'].astype('int8')

#setting network parameters
n_hidden1 = 10
n_hidden2 = 5

n_input = 10
n_output = 1

#setting learning parameters 
learning_constant = 0.02
number_epochs = 4000
batch_size = 189

#weights and biases, input and output settings
X = tf.placeholder(tf.float32, [None, n_input])
Y = tf.placeholder(tf.float32, [None, n_output])

b1 = tf.Variable(tf.random_normal([n_hidden1]))
b2 = tf.Variable(tf.random_normal([n_hidden2]))
b3 = tf.Variable(tf.random_normal([n_output]))

w1 = tf.Variable(tf.random_normal([n_input, n_hidden1]))
w2 = tf.Variable(tf.random_normal([n_hidden1, n_hidden2]))
w3 = tf.Variable(tf.random_normal([n_hidden2, n_output]))

#defining batch input and output data
batch_x1=frog_features['SR']
batch_x2=frog_features['TR']
batch_x3=frog_features['VR']
batch_x4=frog_features['SUR1']
batch_x5=frog_features['SUR2']
batch_x6=frog_features['UR']
batch_x7=frog_features['FR']
batch_x8=frog_features['OR']
batch_x9=frog_features['RR']
batch_x10=frog_features['BR']

batch_y1=frog_labels['Brown frogs']

label=batch_y1

batch_x=np.column_stack((batch_x1, batch_x2, batch_x3,batch_x4,batch_x5,batch_x6,batch_x7,batch_x8,batch_x9,batch_x10))
batch_y = np.column_stack((batch_y1))
batch_y = np.transpose(batch_y)

batch_x_train=batch_x[0:100]
batch_y_train = batch_y[0: 100]

batch_x_test=batch_x[100:188]
batch_y_test=batch_y[100:188]

label_train=label[0:100]
label_test=label[101:128]

def multilayer_perceptron(input_d):
  #Task of neurons of first hidden layer
  layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(input_d,  w1), b1))
  #Task of neurons of second hidden layer
  layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1,  w2), b2))
  #Task of neurons of output layer
  out_layer = tf.add(tf.matmul(layer_2,  w3), b3)

  return out_layer

#Create model
neural_network = multilayer_perceptron(X)

#Define loss and optimizer
loss_op = tf.reduce_mean(tf.math.squared_difference(neural_network,Y))
optimizer = tf.train.GradientDescentOptimizer(learning_constant).minimize(loss_op)

#Initializing the variables
init = tf.global_variables_initializer()

#running session
with tf.Session() as sess:
  sess.run(init)

  #Training epoch
  for epoch in range(number_epochs):
    sess.run(optimizer, feed_dict={X: batch_x_train, Y:batch_y_train})
  
  # Test model using test data
  pred = neural_network
  prediction = pred.eval({X: batch_x_test})

  #accuracy=tf.keras.losses.MSE(pred,Y)
  #print("Accuracy:", accuracy.eval({X: batch_x_test, Y:batch_y_test}))

  #setting a threshold so predicted values can be converted to labels 0 or 1 
  for value in prediction:
    if value[0] < 0.5:
      value[0] = 0
    else:
      value[0] = 1   
  

  #results of actual v.s. predicted. 
  plt.plot(batch_y_test, 'ro', prediction, 'bo')
  plt.ylabel('labels (0,1)')
  plt.title('Expected(red) vs Predicted(blue) results')
  plt.show()
  
  label_list = label.to_list()
  correct_prediction = tf.equal(prediction,label_list)
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  print("Final accuracy result: ", accuracy.eval({X: batch_x_test}))
  print()

  #CROSS VALIDATION

  print("Performing 10th fold cross vaidation")

  #randomly splitting dataset into 10 folds
  c = list(zip(batch_x, batch_y))
  random.shuffle(c)
  batch_x_validate, batch_y_validate = zip(*c)

  x_folds = np.array_split(np.array(batch_x_validate), 10)
  y_folds = np.array_split(np.array(batch_y_validate), 10)
  
  sum = 0
  
  #testing for i'th fold while other folds act as training data
  for i in range(10):
    x_validation_test = x_folds[i]
    y_validation_test = y_folds[i]

    x_validation_train = []
    y_validation_train = []

    for j in range(10):
      if j != i:
        for fold in x_folds:
          for f in range(len(fold)):
            data = fold[f]
            x_validation_train.append(data)
        for fold in y_folds:
          for r in range(len(fold)):
            data = fold[f]
            y_validation_train.append(data)

    #training the 9 folds that are not i'th fold
    for epoch in range(number_epochs):
      sess.run(optimizer, feed_dict={X:x_validation_train, Y:y_validation_train})
    
    #checking results on the validation test data
    pred = neural_network
    output=pred.eval({X: x_validation_test})

    for value in output:
      if value[0] < 0.5:
        value[0] = 0
      else:
        value[0] = 1 

    v_correct_prediction = tf.equal(output, y_validation_test)
    v_accuracy = tf.reduce_mean(tf.cast(v_correct_prediction, tf.float32))
    print("accuracy when testing on fold", i, ": ",  v_accuracy.eval({X: x_validation_test}))
