# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 18:33:05 2019

@author: Xiao
"""
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from pandas import DataFrame
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
import pandas as pd
from collections import Counter
from sklearn.utils import resample
from sklearn.metrics import roc_auc_score

f = open ( "user-item-rating-dataset.csv" , 'r')
l = []

l = [ line.strip().split(",") for line in f]
dataset = np.array(l)
df = DataFrame(dataset)

df.iloc[724,:]
df.iloc[0,:]

################# Exploratory Data Analysis ###################################
"convert to binary classification problem"
rating_column = df[12].values
rating_column_int =  rating_column.astype(int)
df[12] = [1 if r>=4 else 0 for r in rating_column_int]
df[12].value_counts()


################# deal with imbalanced data ##################################
# Separate majority and minority classes
df_majority = df[df[12]==1]
df_minority = df[df[12]==0]

# Upsample minority class
seed = 555
df_minority_upsampled = resample(df_minority, 
                                 replace=True,     # sample with replacement
                                 n_samples=len(df_majority),    # to match majority class
                                 random_state=seed) # reproducible results

df_upsampled = pd.concat([df_majority, df_minority_upsampled])

# Display new class counts
df_upsampled[12].value_counts()

y = df_upsampled[12].values
X = df_upsampled.drop([0,1,12],axis = 1).values

writeFile = "df_upsampled.csv"
df_upsampled.to_csv(writeFile,index = False)

################## tuning the model by modifying number of K ##################
"create test dataset with stratified sampling on y"
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=seed, stratify=y)
# Setup arrays to store train and test accuracies
neighbors = np.arange(1, 40)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

# Loop over different values of k
for i, k in enumerate(neighbors):
    # Setup a k-NN Classifier with k neighbors: knn
    knn = KNeighborsClassifier(n_neighbors=k)

    # Fit the classifier to the training data
    knn.fit(X_train, y_train)

    
    #Compute accuracy on the training set
    cv_results = cross_val_score(knn,X_train,y_train,cv = 5)
    train_accuracy[i] = np.mean(cv_results)

    #Compute accuracy on the testing set
    cv_results = cross_val_score(knn,X_test,y_test,cv = 10)
    test_accuracy[i] = knn.score(X_test, y_test)

# Generate plot
plt.title('k-NN: Varying Number of Neighbors')
plt.plot(neighbors, test_accuracy, label = 'Testing Accuracy')
plt.plot(neighbors, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()

#################### cross validation on the dataset ##########################
#cv_results = cross_val_score(knn,X,y,cv = 5)
#print(cv_results)
#
#np.mean(cv_results)


################## KNN classification ########################################
"create classifier with number of neighbors of 6"
knn = KNeighborsClassifier(n_neighbors = 6)

"fit the classifier with training data"
knn.fit(X,y)
#X_train[0].reshape(1,-1).shape
#distance,ind = knn.kneighbors(X_train[0].reshape(1, -1))
#X_train[724]

"get the accurary on test data"
cv_results = cross_val_score(knn,X,y,cv = 5)
np.mean(cv_results)
predictions = knn.predict(X)

counter = Counter(predictions)


## Predict class probabilities
#prob_y_1 = knn.predict_proba(X)
#
## Keep only the minority class
#prob_y_2 = [p[0] for p in prob_y_1]
#
#print( roc_auc_score(y, prob_y_2) )


############################## Bagging ########################################
from sklearn import model_selection
from sklearn.ensemble import BaggingClassifier

seed = 7
kfold = model_selection.KFold(n_splits=10, random_state=seed)
num = 100
model = BaggingClassifier(base_estimator=knn, n_estimators=num, random_state=seed)
model.fit(X_train, y_train)
results = cross_val_score(model, X_test, y_test, cv=kfold)
print(results.mean())
predictions_bagging = model.predict(X)
counter_bag = Counter(predictions_bagging)

"save the model"
from sklearn.externals import joblib
# Output a pickle file for the model
joblib.dump(model, 'bagging_model.pkl') 

##################################### AUC #####################################

import sklearn.metrics as metrics
# calculate the fpr and tpr for all thresholds of the classification
probs = model.predict_proba(X_test)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
roc_auc = metrics.auc(fpr, tpr) 

# method I: plt
import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

## method II: ggplot
#from ggplot import *
#df = pd.DataFrame(dict(fpr = fpr, tpr = tpr))
#ggplot(df, aes(x = 'fpr', y = 'tpr')) + geom_line() + geom_abline(linetype = 'dashed')