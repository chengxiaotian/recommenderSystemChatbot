# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 18:08:08 2019

@author: Xiao

@INPUT:
    dataset: read-in, raw data to be matrix factorized
    import in MF package
@OUTPUT:
    the final matrices P and Q
"""
import matplotlib.pyplot as plt
from pandas import DataFrame
from sklearn.metrics.pairwise import cosine_similarity
import numpy
import MF0 as mf
import Evaluation as ev



f = open ( "data.csv" , 'r')
l = []
l = [ line.strip().split(",") for line in f]
R = numpy.array(l,numpy.float64)


R = numpy.array(R)

N = len(R)
M = len(R[0])

################gradiant search for best neighborhood number###################
"gradiant search"
#mf.gradient_search(N,M,R)

################ matrix factorization with best number ########################

"do matrix factorization with feature number equals 5"
"use 5 features to initiate matrix factorization"
#k = 5
#P = numpy.random.rand(N,k)
#Q = numpy.random.rand(M,k)
#nP, nQ = mf.matrix_factorization(R, P, Q, k)
#
#error_rate = ev.getErrorRate(R,numpy.dot(nP,nQ.T))
#        
#"imputed rate matrix"
#R_new =  numpy.dot(nP,nQ.T).T
#R_new.shape
#
#numpy.amax(R_new)
#numpy.amin(R_new)


################ collaborative filering #######################################
f = open ( "imputed_mx.csv" , 'r')
l = []
l = [ line.strip().split(",") for line in f]
R_new = numpy.array(l,numpy.float64)   


df = DataFrame(R_new)
rating_mx_origin = R.T

"get new rating matrix by deducting the user mean"
##get user mean list
#userMean_all = []
#for i in range(len(rating_mx_origin)):
#    userMean = []
#    for j in range(len(rating_mx_origin[i])):
#        if rating_mx_origin[i][j]>0:
#            userMean.append(rating_mx_origin[i][j])
#        np1 = numpy.array(userMean)
#        user_mean = numpy.mean(np1)
#    userMean_all.append(user_mean)
#
##get matrix deducting user mean from original data
#df_new = DataFrame()
#for i in range(len(df)):
#    arrRow = []
#    for d in df.iloc[i]:
#        newD = d - userMean_all[i]
#        arrRow.append(newD)
#        row_df = DataFrame([arrRow])
#    df_new = df_new.append(row_df)
#    
"similarity matrix - nonnormalized rating table - for comparison purpose" 
#similarity = numpy.zeros((168, 168))
#
#for i in range(len(similarity)):
#    for j in range(len(similarity[i])):
#        similarity[i][j] = cosine_similarity([df[i].values], [df[j].values])
        
"similarity matrix - normalized rating table"
f = open ( "df_new.csv" , 'r')
l = []
l = [ line.strip().split(",") for line in f]
df_new1 = numpy.array(l,numpy.float64)
df_new = DataFrame(df_new1)

similarity_normalized = numpy.zeros((168, 168))

for i in range(len(similarity_normalized)):
    for j in range(len(similarity_normalized[i])):
        similarity = cosine_similarity([df_new[i].values], [df_new[j].values])
        if similarity>0:
            similarity_normalized[i][j] = similarity
        else:
            similarity_normalized[i][j] = 0
            
"calculate weighted rating basing on neighborhoods"
predict_rate_cf_np = numpy.zeros((158,168))
for user_index in range(len(rating_mx_origin)):
    for item_index in range(len(rating_mx_origin[user_index])):
        sumproduct_rate_sim = 0
        sumweight = 0
        simRateProductList = []
        weightList = []
        for sim_index in range(len(similarity_normalized[item_index])):
            if rating_mx_origin[user_index][sim_index] and similarity_normalized[item_index][sim_index] != 1>0:
                simRateProductList.append(rating_mx_origin[user_index][sim_index]*similarity_normalized[item_index][sim_index])
                weightList.append(similarity_normalized[item_index][sim_index])
        
        #get top 5 weights and the index
        weight_np = numpy.array(weightList)
        topWeightList_index = weight_np.argsort()[::-1][:5]
        
        for ind in topWeightList_index:
            sumproduct_rate_sim = sumproduct_rate_sim + simRateProductList[ind]
            sumweight = sumweight + weightList[ind]
            
        predict_rate = sumproduct_rate_sim/sumweight
        predict_rate_cf_np[user_index][item_index] = predict_rate

"evaluation base on error rate"
error_rate2 = ev.getErrorRate(rating_mx_origin,predict_rate_cf_np)


######################## output computed matrix to file ########################
"write factorized matrix into file"
#writeFile = "imputed_mx.csv"
#with open(writeFile,'w') as file:
#    for row in R_new:
#        writeline = ",".join([str(r) for r in row])
#        file.write(writeline)
#        file.write("\n")
        
#"write detailed information into file - items features"
#writeFile = "nP-item.csv"
#with open(writeFile,'w') as file:
#    for row in nP:
#        writeline = ",".join([str(r) for r in row])
#        file.write(writeline)
#        file.write("\n")
#        
#"write detailed information into file - user features"
#writeFile = "nQ-user.csv"
#with open(writeFile,'w') as file:
#    for row in nQ:
#        writeline = ",".join([str(r) for r in row])
#        file.write(writeline)
#        file.write("\n")
#
#"write normalized rating matrix into file"
#writeFile = "df_new.csv"
#with open(writeFile,'w') as file:
#    for row in df_new.values:
#        writeline = ",".join([str(r) for r in row])
#        file.write(writeline)
#        file.write("\n")

#"write predicted rating matrix with CF into file"
#writeFile = "predict_rate_cf_np.csv"
#with open(writeFile,'w') as file:
#    for row in predict_rate_cf_np:
#        writeline = ",".join([str(r) for r in row])
#        file.write(writeline)
#        file.write("\n")
