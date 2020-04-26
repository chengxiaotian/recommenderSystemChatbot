# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 18:28:57 2019

@author: Xiao
"""
import numpy

def getErrorRate(rating_mx_origin,predict_rate_mx):
    
    "evaluation base on error rate"
    count = 0
    eij = 0
    for i in range(len(rating_mx_origin)):
        for j in range(len(rating_mx_origin[i])):
            if rating_mx_origin[i][j] > 0:
                count = count +1
                eij = eij + pow(rating_mx_origin[i][j]/10 - numpy.nan_to_num(predict_rate_mx[i][j]/10),2)
    
    error_rate_cf = numpy.sqrt(eij/count)
    return error_rate_cf