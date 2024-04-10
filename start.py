#Cammer and Singer SVM


import numpy as np
from methods import SVM_CS
from kernels import Polynomial
from extractors import HOG
import pandas as pd

from data_loader import X,Y,X_test

extractor = HOG(4,9,'sobel')
kernel = Polynomial(100,0,4)

print("compute K matrices")

Xtrain=extractor.transform(X)
Xtest=extractor.transform(X_test)

K_train=kernel.gram(Xtrain)
K_test=kernel.matrix(Xtest,Xtrain)

method=SVM_CS(lamda=0.1)
print("fit")
method.fit(K_train,Y,timeout=None,n_iter=10)
print("predict")
Y_hat=method.predict(K_test)

dataframe = pd.DataFrame({'Prediction' : Y_hat})
dataframe.index += 1
dataframe.to_csv('Yte_pred.csv',index_label='Id')
