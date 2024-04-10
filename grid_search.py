f_results="./results/"
algo="1v1"
ker="tanh"

print(algo)

f_logs=f_results+ker+"/"+algo+"/logs/"
f_accs=f_results+ker+"/"+algo+"/accs/"
f_hist=f_results+ker+"/"+algo+"/hist"
f_k=f_results+ker+"/"+algo+"/k"

C=10

import pickle
import json
import numpy as np
from itertools import product
import pandas as pd
import os
from time import time
import random


from methods import SVM_1vR, SVM_1v1, SVM_CS
from kernels import Polynomial, RBF, Tanh
from extractors import Id, Pooling, HOG

from data_loader import X_train,Y_train,X_val,Y_val
from utils import *

crop=None

if crop is not None :
    X_train=X_train[:crop]
    Y_train=Y_train[:crop]

#load historic (i.e. tried hyperparameters) and current id k
with open(f_hist, 'rb') as f :
    hist=pickle.load(f)
with open(f_k, 'r') as f :
    k=json.load(f)

if algo=='1v1' :
    timeout=5*60
elif algo=='1vR' :
    timeout=60*60
elif algo=='CS' :
    timeout=None

gammas=[0.1,1,10,100]
rs=[-1,0,1]
ds=[2,3,4]
extractors=[None,("max",3,2),('min',3,2),('mean',3,2),\
	('hog',4,9,'sobel'),('hog',6,9,'sobel'),('hog',8,9,'sobel'),\
    ('hog',4,9,'classic'),('hog',6,9,'classic'),('hog',8,9,'classic')]
lamdas=[0.1,0.5,1,5,10]

if ker == "poly" :
    list_parameters = list(product(gammas,rs,ds,extractors,lamdas))
elif ker == "rbf" :
    list_parameters = list(product(gammas,extractors,lamdas))
elif ker == "tanh" :
    list_parameters = list(product(gammas,rs,extractors,lamdas))

random.shuffle(list_parameters)
for tupl in list_parameters :
    print(k,tupl)
    if tupl in hist :
        print("Already done")
    else :
        if ker == "poly" :
            gamma,r,d,extractor,lamda=tupl
            kernel=Polynomial(gamma,r,d)

        elif ker == "rbf" :
            gamma,extractor,lamda=tupl
            r=d=None
            kernel=RBF(gamma)

        elif ker == "tanh" :
            gamma,r,extractor,lamda=tupl
            d=None
            kernel=Tanh(gamma,r)

        #Construct K matrices
        if extractor is None :
            transformer=Id()
        elif extractor[0] in ['max','min','mean'] :
            transformer=Pooling(*extractor)
        elif extractor[0]=='hog' :
            transformer=HOG(*extractor[1:])

        Xtr=transformer.transform(X_train)
        Xva=transformer.transform(X_val)

        K_train=kernel.gram(Xtr)
        K_val=kernel.matrix(Xva,Xtr)

        #Construct kernel method
        if algo == "1vR" :
            method=SVM_1vR(lamda=lamda)
        elif algo == "1v1" :
            method=SVM_1v1(lamda=lamda)
        elif algo == "CS" :
            method=SVM_CS(lamda=lamda)

        t0=time()
        #test if the method does not time out
        if method.fit(K_train,Y_train,timeout) :
            t=int(time()-t0)

            Y_hat=method.predict(K_val)
            acc=(Y_hat==Y_val).mean()
            print(acc)
            if algo == "1v1" :
                #compute sum prediction
                Y_hat=method.predict(K_val,method="sum")
                acc_sum=(Y_hat==Y_val).mean()

                #compute accurcay of each pair of classes for vote prediction
                values=method.regress(K_val)
                accuracies={}
                for c1 in range(C) :
                    for c2 in range(c1+1,C) :
                        mask=(Y_val==c1) | (Y_val==c2)
                        yc=Y_val[mask]
                        vc=values[mask,c1,c2]
                        p=((vc>0) & (yc==c1)).sum() + ((vc<0) & (yc==c2)).sum()
                        accuracies[(c1,c2)]=p/mask.sum()
                with open(f_accs+str(k), 'wb') as f :
                    pickle.dump(accuracies,f)
            else :
                acc_sum=-1
        else :
            print("TIMEOUT")
            t=int(time()-t0)
            acc=acc_sum=-1
        log={'gamma':gamma,'r':r,'d':d,'extractor':extractor,'lamda':lamda,\
                'acc':acc,'acc_sum':acc_sum,'t':t}

        #save log, historic, current id
        with open(f_logs+str(k), 'w') as f :
            json.dump(log,f)
        hist.add(tupl)
        with open(f_hist, 'wb') as f :
            pickle.dump(hist,f)
        k+=1
        with open(f_k, 'w') as f :
            json.dump(k,f)



