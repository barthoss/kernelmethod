import numpy as np
import scipy
from qpsolvers import solve_qp

C=10

class Method() :
    def __init__(self,*args, **kwargs) :
        pass
    def fit(self,K,y,*args, **kwargs) :
        """
        Input: 
           K: gram matrix of the training set
           y: classes of the training set
        Output:
           True if the method fitted
           False if the method fitting timed out
        """
        pass
    def predict(self,*args, **kwargs) :
        """
        Input:
           K: similarity matrix between training set and test set
        Outpu:
           Prediction of classes
        """
        pass

class DistSet(Method) :
    """
    Classify an image with the nearest centroid of classes
    """
    def fit(self,K,y) :
        self.K=K
        self.y=y
        return True

    def predict(self,K_val,K_normval) :
        # Use formula in part 4 of the report
        n=K_normval.size
        dist=np.zeros((n,C))
        
        dist+=K_normval[:,None]
        for c in range(C) :
            dist[:,c]-=2*K_val[:,self.y==c].mean(axis=1)
        for c in range(C) :
            mask=self.y==c
            dist[:,c]+=self.K[mask,:][:,mask].mean()
        prediction=np.argmin(dist,axis=1)

        return prediction
    
class KNN(Method) :
    """
    Performs k-NN classification
    """
    def __init__(self,k=10) :
        self.k=k

    def fit(self,K,y) :
        self.y=y
        return True

    def predict(self,K) :
        n=K.shape[0]
        idx=np.argpartition(K,self.k,axis=-1)
        idx=idx[:,:self.k]

        knn_classes=self.y[idx]
        knn_count=np.zeros((n,C))
        for c in range(C) :
            knn_count[:,c]=(knn_classes==c).sum(axis=1)

        return knn_count.argmax(axis=1)
    
class SVM_bin(Method) :
    """
    Performs a binary SVM with labels -1 and 1
    """
    def __init__(self,lamda) :
        self.lamda=lamda

    def fit(self,K,y,time_limit=None) :
        n=K.shape[0]
        c=1/(2*self.lamda*n)
        G=scipy.sparse.vstack([-scipy.sparse.diags(y),
                               scipy.sparse.diags(y)])
        h=np.hstack([np.zeros(n),c*np.ones(n)])

        alpha0=2*c*np.random.rand(n)-c

        if time_limit is None :
            self.alpha=solve_qp(P=K,q=-y,G=G,h=h, initvals=alpha0, solver="clarabel")
        else :
            self.alpha=solve_qp(P=K,q=-y,G=G,h=h, initvals=alpha0, solver="clarabel",time_limit=time_limit)

        return self.alpha is not None
            
    def predict(self,K) :
        return np.sign(self.regress(K))
    def regress(self,K) :
        #we only compute on the support vector
        mask=self.alpha!=0
        return K[:,mask]@self.alpha[mask]

class SVM_1vR(Method) :
    """
    Performs 1vR SVM using binary SVM
    """
    
    def __init__(self,lamda) :
        self.svms=[]
        for c in range(C) :
            self.svms.append(SVM_bin(lamda))

    def fit(self,K,y,time_limit=None) :
        K=scipy.sparse.csc_matrix(K)
        for c in range(C) :
            yc=2*(y==c).astype(int)-1
            fitted=self.svms[c].fit(K,yc,time_limit)
            if not fitted :
                return False
        return True

    def regress(self,K) :
        """
        Input:
          K [n_test,n_train]: similarity matrix
        Ouput:
          [C,n_test]: SVM value for each pair class/sample
        """
        return np.vstack([svm.regress(K) for svm in self.svms])

    def predict(self,K) :
        values = self.regress(K)
        prediction = np.argmax(values,axis=0)
        return prediction

    
class SVM_1v1(Method) :
    """
    Performs 1vR SVM using binary SVM
    Here the lamda and K can be a dict
       if we want different kernel/hyperparameter for each class
    """
    def __init__(self,lamda) :
        self.svms={}
        for c1 in range(C) :
            for c2 in range(c1+1,C) :
                if type(lamda)==dict :
                    l=lamda[(c1,c2)]
                else :
                    l=lamda
                self.svms[(c1,c2)]=SVM_bin(l)

    def fit(self,K,y,time_limit=None) :
        if type(K)!=dict :
            K=scipy.sparse.csc_matrix(K)
        self.y=y
        for c1 in range(C) :
            for c2 in range(c1+1,C) :
                mask=(y==c1) | (y==c2)
                if type(K)!=dict :
                    Kc=K[mask,:][:,mask]
                else :
                    Kc=scipy.sparse.csc_matrix(K[(c1,c2)])
                yc=y[mask]
                yc=2*(yc==c1).astype(int)-1
                fitted=self.svms[(c1,c2)].fit(Kc,yc,time_limit)
                if not fitted :
                    return False
        return True

    def regress(self,K) :
        """
        Input:
          K [n_test,n_train]: similarity matrix
        Ouput:
          [C,C,n_test]: SVM value for each triplet class/class/sample
        """
        if type(K)!=dict :
            K=scipy.sparse.csc_matrix(K)
            n=K.shape[0]
        else :
            n=K[(0,1)].shape[0]
        values=np.zeros((n,C,C))
        for c1 in range(C) :
            for c2 in range(c1+1,C) :
                mask=(self.y==c1) | (self.y==c2)
                if type(K)!=dict :
                    Kc=K[:,mask]
                else :
                    Kc=scipy.sparse.csc_matrix(K[(c1,c2)])
                v=self.svms[(c1,c2)].regress(Kc)
                values[:,c1,c2]=v
                values[:,c2,c1]=-v
        return values
    def predict(self,K,method="vote") :
        values = self.regress(K)
        if method=="vote" :
            values=(values>0).astype(float)
        elif method!="sum" :
            raise Exception("method must be vote or sum")
        values=values.sum(axis=-1)
        prediction = np.argmax(values,axis=-1)
        
        return prediction

class SVM_CS(Method) :
    """
    Performs Crammer & Singer SVM as in https://jmlr.csail.mit.edu/papers/volume2/crammer01a/crammer01a.pdf
    """
    def __init__(self,lamda) :
        self.lamda=lamda
    def fit(self,K,y,timeout,n_iter=2e5) :
        n=len(K)
        self.alpha=np.zeros((C,n))
        F=np.zeros((C,n))
        for c in range(C) :
            F[c]-=(y==c)*self.lamda

        for _ in range(int(n_iter)) :
            #chose item to optimize
            mask=np.zeros((C,n))
            for c in range(C) :
                mask[c]=(self.alpha[c]>=(y==c))
            phi=F.max(axis=0)-np.ma.masked_array(F,mask).min(axis=0).data
            p=phi.argmax()

            #solve QP
            b=self.alpha@K[p]-K[p,p]*self.alpha[:,p]
            b[y[p]]-=self.lamda
            h=np.zeros(C)
            h[y[p]]=1
            alpha_star = solve_qp(P=K[p,p]*np.eye(C),
                            q=b,
                            G=scipy.sparse.eye(C),
                            h=h,
                            A=np.ones((1,C)),
                            b=np.array([0]),
                            solver="proxqp")
            
            #update values
            Dalpha=alpha_star-self.alpha[:,p]

            F+=Dalpha[:,None]@K[p][None,:]
            self.alpha[:,p]=alpha_star
        return True
    def regress(self,K) :
        return self.alpha@K.T
    
    def predict(self,K) :
        values = self.regress(K)
        return values.argmax(axis=0)
