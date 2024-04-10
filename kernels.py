import numpy as np

class Kernel () :
    def __init__ (self) :
        pass
    def __call__ (self,x,y) :
        """
        Compute the piece wise similarity between elemnts
        """
        pass
    def matrix(self,X,Y) :
        """
        Compute the similarity matrix
        between two sets of points
        """
        pass
    def gram(self,X) :
        """
        Compute the gram matrix
        of a set of points
        """
        return self.matrix(X,X)


class Linear(Kernel) :
    def __init__ (self) :
        pass
    def __call__ (self,X,Y) :
        return (X*Y).sum(axis=1)
    def matrix(self,X,Y) :
        return X@Y.T
    
class RBF(Kernel) :
    def __init__ (self,gamma) :
        self.gamma=gamma
    def __call__ (self,X,Y) :
        Ds=((X-Y)**2).sum(axis=-1)
        return np.exp(-self.gamma*Ds)
    def matrix(self,X,Y) :
        X2=(X**2).sum(axis=-1)
        Y2=(Y**2).sum(axis=-1)
        Ds=X2[:,None]+Y2[None,:]-2*X@Y.T
        return np.exp(-self.gamma*Ds)

class Polynomial(Kernel) :
    def __init__ (self,gamma,r,d) :
        self.gamma=gamma
        self.r=r
        self.d=d
    def __call__ (self,X,Y) :
        return (self.gamma*(X*Y).sum(axis=1)+self.r)**self.d
    def matrix(self,X,Y) :
        return (self.gamma*X@Y.T+self.r)**self.d
    
class Tanh(Kernel) :
    def __init__ (self,gamma,r) :
        self.gamma=gamma
        self.r=r
    def __call__ (self,X,Y) :
        return np.tanh(self.gamma*(X*Y).sum(axis=1)+self.r)
    def matrix(self,X,Y) :
        return np.tanh(self.gamma*X@Y.T+self.r)
    
