# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 18:06:08 2019

@author: Xiao
"""

'''
Neural Network
'''

import numpy
import pandas as pd
from pandas import DataFrame
import keras
from keras.layers import Dense
from keras.models import Sequential
from sklearn.preprocessing import StandardScaler
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical 
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from collections import Counter
from sklearn import model_selection
from sklearn.model_selection import ShuffleSplit
from keras.utils import plot_model
import matplotlib.pyplot as plt

"""
imbalance data:
1. upscale sample
2. downscale sample


dataset too small:
1. cross validation
2. bagging
3. leave-one-out sampling

plot accuracy rate
"""


f = open ( "poi-user-rating-dataset1-new.txt" , 'r',encoding = 'utf8')
l = []
l = [ line.strip().split("\t") for line in f]
data = numpy.array(l) 

columns = data[0,1:].tolist()
df = DataFrame(data[1:,1:],columns = columns)
df_train = df.iloc[:,2:]


"convert to binary classification problem"
rating_column = df_train['rating'].values
rating_column_int =  rating_column.astype(int)
df_train.loc[:,'rating'] = [1 if r>=40 else -1 for r in rating_column_int]
df_train['rating'].value_counts()

"change to numeric"
df_train_numeric = df_train.apply(pd.to_numeric)
desc_1 = df_train_numeric.describe()

df_train_numeric['rating'].value_counts()

#predictors = df_train_numeric.iloc[:,1:]
#target = df_train_numeric.iloc[:,0]

X = df_train_numeric.iloc[:,1:].values
y = df_train_numeric.iloc[:,0].values



####################### build network model for logistic regression (single neuron) ###################################
"convert to categorical"
target_cat = to_categorical(y)

"Save the number of columns in predictors: n_cols"
n_cols = X.shape[1]


"Set up the model: model"
model3 = Sequential()

"Add the output layer"
model3.add(Dense(2,activation='softmax',input_shape=(n_cols,)))

"fitting the model, with adam optimizer, and mean squared error loss function"
model3.compile(optimizer = 'adam',loss = 'categorical_crossentropy',metrics=['accuracy'])

"Define early_stopping_monitor"
early_stopping_monitor = EarlyStopping(patience = 3)

"Fit the model"
model_training = model3.fit(X,target_cat,epochs = 30,validation_split = 0.3,callbacks = [early_stopping_monitor])
predictions = model3.predict(X)

model3.summary()


############### build network model for logistic regression (more layers and neurons) ###################################
"Save the number of columns in predictors: n_cols"
n_cols = X.shape[1]

seed = 7
numpy.random.seed(seed)

# define 10-fold cross validation test harness
kfold = ShuffleSplit(n_splits=10, random_state=seed)
cvscores1 = []
y = target_cat[:,1]
for train, test in kfold.split(X,y):
    
    model = Sequential()
    model.add(Dense(10, input_shape=(n_cols,), activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
    
    early_stopping_monitor = EarlyStopping(patience = 3)
    model_training = model.fit(X[train], y[train], epochs=150, validation_split = 0.2,batch_size=10,callbacks = [early_stopping_monitor])
    scores = model.evaluate(X[test], y[test])
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    cvscores1.append(scores[1] * 100)
print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores1), numpy.std(cvscores1)))

#predictions = model.predict(X)

# Create the plot for the accuracy for each validation
plt.plot(cvscores1, 'r')
plt.xlabel('cv')
plt.ylabel('cv scores')
plt.show()

# Create the plot 
plt.plot(model_training.history['loss'], 'b',label = 'Training Loss')
plt.plot(model_training.history['val_loss'], 'r',label = 'Testing Loss')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Validation score')
plt.show()

model.summary()

# Plot the model
plot_model(model, to_file='model.png')

# Display the image
data = plt.imread('model.png')
plt.imshow(data)
plt.show()

model.save('neural_network.h5')


################## build neural network comparison model(more neurons) ######################

seed = 7
numpy.random.seed(seed)

# define 10-fold cross validation test harness
kfold = ShuffleSplit(n_splits=10, random_state=seed)
cvscores2 = []
for train, test in kfold.split(X,target_cat):
    
    model2 = Sequential()
    model2.add(Dense(1000, input_shape=(n_cols,), activation='relu'))
    model2.add(Dense(1000, activation='relu'))
    model2.add(Dense(1, activation='sigmoid'))
    
    model2.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
    
    early_stopping_monitor = EarlyStopping(patience = 3)
    model_training = model2.fit(X[train], y[train], epochs=150, validation_split = 0.2,batch_size=10,callbacks = [early_stopping_monitor])
    scores = model2.evaluate(X[test], y[test])
    print("%s: %.2f%%" % (model2.metrics_names[0], scores[0]*100))
    cvscores2.append(scores[1] * 100)
print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores2), numpy.std(cvscores2)))

#predictions = model2.predict(X)


# Create the plot for the accuracy for each validation
plt.plot(cvscores2, 'r')
plt.xlabel('cv')
plt.ylabel('cv scores')
plt.show()

# Create the plot 
plt.plot(model_training.history['loss'], 'b',label = 'Training Loss')
plt.plot(model_training.history['val_loss'], 'r',label = 'Testing Loss')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Validation score')
plt.show()

################## build neural network comparison model (more layers) ######################

seed = 7
numpy.random.seed(seed)

# define 10-fold cross validation test harness
kfold = ShuffleSplit(n_splits=10, random_state=seed)
cvscores3 = []
for train, test in kfold.split(X,y):
    
    model3 = Sequential()
    model3.add(Dense(10, input_shape=(n_cols,), activation='relu'))
    model3.add(Dense(10, activation='relu'))
    model3.add(Dense(10, activation='relu'))
    model3.add(Dense(10, activation='relu'))
    model3.add(Dense(10, activation='relu'))
    model3.add(Dense(1, activation='sigmoid'))
    
    model3.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
    
    early_stopping_monitor = EarlyStopping(patience = 3)
    model_training = model3.fit(X[train], y[train], epochs=150, validation_split = 0.2,batch_size=10,callbacks = [early_stopping_monitor])
    scores = model3.evaluate(X[test], y[test])
    print("%s: %.2f%%" % (model2.metrics_names[1], scores[1]*100))
    cvscores3.append(scores[1] * 100)
print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores3), numpy.std(cvscores3)))


# Create the plot for the accuracy for each validation
plt.plot(cvscores3, 'r')
plt.xlabel('cv')
plt.ylabel('cv scores')
plt.show()

# Create the plot 
plt.plot(model_training.history['loss'], 'b',label = 'Training Loss')
plt.plot(model_training.history['val_loss'], 'r',label = 'Testing Loss')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Validation score')
plt.show()

predictions = model3.predict(X)



############# build network model with leave-one-out sampling method ##########

from sklearn.model_selection import LeaveOneOut
loo = LeaveOneOut()

cvscores = []
for train, test in loo.split(X):
    
    model = Sequential()
    model.add(Dense(12, input_shape=(n_cols,), activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
    
    early_stopping_monitor = EarlyStopping(patience = 3)
    model_training = model.fit(X[train], y[train], epochs=150, validation_split = 0.2,batch_size=10,callbacks = [early_stopping_monitor])
    scores = model.evaluate(X[test], y[test])
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    cvscores.append(scores[1] * 100)
print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))

