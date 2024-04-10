data_folder = "./data/"
import pandas as pd
import numpy as np
C=10
p_train=0.8

#Opening training files
X=pd.read_csv(data_folder+'Xtr.csv', sep=',', header=None).values
X=X[:,:-1]
X=X.reshape((5000,32,32,3),order='F')
Y=pd.read_csv(data_folder+'Ytr.csv', sep=',').values
Y=Y[:,1]


#Split train/val to have each class at the same proportion
np.random.seed(0)

args=np.arange(len(Y))
np.random.shuffle(args)
X=X[args]
Y=Y[args]

args_train=[]
args_eval=[]
for c in range(C) :
    argsc=np.where(Y==c)[0]
    n_train=int(p_train*argsc.size)
    args_train.append(argsc[:n_train])
    args_eval.append(argsc[n_train:])
args_train=np.concatenate(args_train)
args_eval=np.concatenate(args_eval)
np.random.shuffle(args_train)
np.random.shuffle(args_eval)

X_train=X[args_train]
Y_train=Y[args_train]
X_val=X[args_eval]
Y_val=Y[args_eval]

X_test=pd.read_csv(data_folder+'Xte.csv', sep=',', header=None).values
X_test=X_test[:,:-1]
X_test=X_test.reshape((2000,32,32,3),order='F')
