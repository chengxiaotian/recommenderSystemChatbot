# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 18:08:08 2019

@author: Xiao
"""

import numpy
import matplotlib.pyplot as plt



###############################################################################

"""
@INPUT:
    R     : a matrix to be factorized, dimension N x M
    P     : an initial matrix of dimension N x K
    Q     : an initial matrix of dimension M x K
    K     : the number of latent features
    steps : the maximum number of steps to perform the optimisation
    alpha : the learning rate
    beta  : the regularization parameter
@OUTPUT:
    the final matrices P and Q
"""
def matrix_factorization(R, P, Q, K, steps=5000, alpha=0.0002, beta=0.02):
    Q = Q.T
    for step in range(steps):
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    eij = R[i][j] - numpy.dot(P[i,:],Q[:,j])
                    for k in range(K):
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
        eR = numpy.dot(P,Q)
        e = 0
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    e = e + pow(R[i][j] - numpy.dot(P[i,:],Q[:,j]), 2)
                    for k in range(K):
                        e = e + (beta/2) * ( pow(P[i][k],2) + pow(Q[k][j],2) )
        if e < 0.001:
            break
    return P, Q.T

################gradiant search for best neighborhood number#################
def gradient_search(N,M,R):
    
    '''
    matrix factorization
    '''
    K_neighbors = numpy.arange(1,10)
    errorRate = []
    
    for index,k in enumerate(K_neighbors):
    
        P = numpy.random.rand(N,k)
        Q = numpy.random.rand(M,k)
        nP, nQ = matrix_factorization(R, P, Q, k)
    
        count = 0
        eij = 0
            
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    count = count +1
                    eij = eij + pow(R[i][j] - numpy.dot(nP[i,:],nQ.T[:,j]),2)
        
        errorRate.append(numpy.sqrt(eij/count))
        
        
        
    errorRate_np = numpy.array(errorRate)
    plt.title('Error rate with the Variance of Selection of K')
    plt.style.use('ggplot')
    plt.plot(K_neighbors, errorRate_np, label = 'error rate')
    plt.legend()
    plt.xlabel('Number of K')
    plt.ylabel('Error Rate')
    plt.show()
    