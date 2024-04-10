#Multi kernel 1v1 voting classifier


import numpy as np
import pandas as pd
import os
from time import time


from methods import SVM_1v1
from kernels import Polynomial, RBF
from extractors import Id, Pooling, HOG

from data_loader import X,Y,X_test
from utils import *

C=10

kers={(0, 1):Polynomial(1,-1,4),
      (0, 2):Polynomial(10,1,2),
      (0, 3):Polynomial(0.1,1,3),
      (0, 4):Polynomial(10,-1,3),
      (0, 5):Polynomial(0.1,1,4),
      (0, 6):Polynomial(1,1,4),
      (0, 7):Polynomial(10,0,2),
      (0, 8):Polynomial(0.1,-1,4),
      (0, 9):Polynomial(1,0,4),
      (1, 2):RBF(0.1),
      (1, 3):Polynomial(10,-1,2),
      (1, 4):Polynomial(1,0,3),
      (1, 5):RBF(1),
      (1, 6):RBF(1),
      (1, 7):RBF(1),
      (1, 8):Polynomial(100,-1,3),
      (1, 9):Polynomial(0.1,-1,3),
      (2, 3):Polynomial(1,1,4),
      (2, 4):Polynomial(1,1,4),
      (2, 5):Polynomial(10,-1,4),
      (2, 6):Polynomial(100,1,3),
      (2, 7):Polynomial(1,0,4),
      (2, 8):Polynomial(10,-1,4),
      (2, 9):Polynomial(100,-1,3),
      (3, 4):Polynomial(1,-1,4),
      (3, 5):Polynomial(0.1,1,4),
      (3, 6):Polynomial(1,0,4),
      (3, 7):RBF(1),
      (3, 8):RBF(1),
      (3, 9):Polynomial(10,0,3),
      (4, 5):Polynomial(1,1,4),
      (4, 6):Polynomial(0.1,-1,3),
      (4, 7):Polynomial(1,1,3),
      (4, 8):Polynomial(10,1,3),
      (4, 9):Polynomial(1,0,4),
      (5, 6):Polynomial(1,0,3),
      (5, 7):RBF(1),
      (5, 8):Polynomial(100,0,3),
      (5, 9):Polynomial(1,-1,4),
      (6, 7):Polynomial(1,0,4),
      (6, 8):Polynomial(10,1,4),
      (6, 9):Polynomial(1,-1,4),
      (7, 8):Polynomial(1,-1,4),
      (7, 9):Polynomial(1,0,2),
      (8, 9):RBF(1)}

extractors={(0, 1):HOG(6,9,'sobel'),
            (0, 2):Pooling('min',3,2),
            (0, 3):HOG(4,9,'classic'),
            (0, 4):Pooling('min',3,2),
            (0, 5):HOG(4,9,'classic'),
            (0, 6):HOG(4,9,'classic'),
            (0, 7):HOG(8,9,'sobel'),
            (0, 8):HOG(4,9,'sobel'),
            (0, 9):HOG(4,9,'sobel'),
            (1, 2):HOG(4,9,'classic'),
            (1, 3):HOG(6,9,'sobel'),
            (1, 4):HOG(8,9,'sobel'),
            (1, 5):HOG(6,9,'sobel'),
            (1, 6):HOG(4,9,'sobel'),
            (1, 7):HOG(6,9,'sobel'),
            (1, 8):HOG(8,9,'sobel'),
            (1, 9):HOG(4,9,'classic'),
            (2, 3):HOG(8,9,'classic'),
            (2, 4):HOG(8,9,'classic'),
            (2, 5):HOG(6,9,'classic'),
            (2, 6):HOG(4,9,'sobel'),
            (2, 7):HOG(8,9,'classic'),
            (2, 8):HOG(6,9,'classic'),
            (2, 9):HOG(6,9,'sobel'),
            (3, 4):HOG(6,9,'classic'),
            (3, 5):HOG(8,9,'classic'),
            (3, 6):HOG(8,9,'classic'),
            (3, 7):HOG(6,9,'sobel'),
            (3, 8):HOG(6,9,'sobel'),
            (3, 9):HOG(6,9,'classic'),
            (4, 5):HOG(6,9,'sobel'),
            (4, 6):HOG(8,9,'classic'),
            (4, 7):HOG(6,9,'sobel'),
            (4, 8):HOG(8,9,'sobel'),
            (4, 9):HOG(4,9,'classic'),
            (5, 6):HOG(8,9,'sobel'),
            (5, 7):HOG(6,9,'classic'),
            (5, 8):HOG(8,9,'classic'),
            (5, 9):HOG(8,9,'sobel'),
            (6, 7):HOG(6,9,'sobel'),
            (6, 8):HOG(8,9,'classic'),
            (6, 9):HOG(8,9,'classic'),
            (7, 8):HOG(8,9,'classic'),
            (7, 9):HOG(6,9,'classic'),
            (8, 9):HOG(6,9,'classic')}
lamdas={(0, 1):0.5,
        (0, 2):0.5,
        (0, 3):0.1,
        (0, 4):5,
        (0, 5):5,
        (0, 6):0.5,
        (0, 7):5,
        (0, 8):0.5,
        (0, 9):10,
        (1, 2):10,
        (1, 3):5,
        (1, 4):1,
        (1, 5):1,
        (1, 6):10,
        (1, 7):1,
        (1, 8):10,
        (1, 9):0.1,
        (2, 3):0.1,
        (2, 4):0.1,
        (2, 5):0.1,
        (2, 6):10,
        (2, 7):10,
        (2, 8):0.1,
        (2, 9):0.5,
        (3, 4):0.5,
        (3, 5):0.1,
        (3, 6):1,
        (3, 7):1,
        (3, 8):1,
        (3, 9):0.1,
        (4, 5):10,
        (4, 6):0.1,
        (4, 7):5,
        (4, 8):5,
        (4, 9):0.1,
        (5, 6):10,
        (5, 7):1,
        (5, 8):1,
        (5, 9):0.5,
        (6, 7):5,
        (6, 8):10,
        (6, 9):1,
        (7, 8):1,
        (7, 9):0.5,
        (8, 9):1}


K_train={}
K_test={}
print("compute K matrices")
for c1 in range(C) :
    for c2 in range(c1+1,C) :
        mask=(Y==c1) | (Y==c2)
        Xtrain=extractors[(c1,c2)].transform(X[mask])
        Xtest=extractors[(c1,c2)].transform(X_test)

        K_train[(c1,c2)]=kers[(c1,c2)].gram(Xtrain)
        K_test[(c1,c2)]=kers[(c1,c2)].matrix(Xtest,Xtrain)

method=SVM_1v1(lamda=lamdas)
print("fit")
method.fit(K_train,Y)
print("predict")
Y_hat=method.predict(K_test)

dataframe = pd.DataFrame({'Prediction' : Y_hat})
dataframe.index += 1
dataframe.to_csv('Yte_pred.csv',index_label='Id')
