import numpy as np
import cv2 as cv


class Extractor:
    def transform(self,X):
        """
        Input: set of rgb images X [batch,n,n,3]
        Output: set of features [batch,k]
        """
        pass

class Id(Extractor) :
    def transform(self,X) :
        batch=X.shape[0]
        return X.reshape(batch,-1,order='F')
    
class Pooling(Extractor) :
    """
    Pool a batch of images
    """
    def __init__ (self,fnct,size,stride) :
        self.size=size
        self.stride=stride
        if fnct=="mean" :
            self.fnct=np.mean
        elif fnct=="max" :
            self.fnct=np.max
        elif fnct=="min" :
            self.fnct=np.min

    def transform(self,X) :
        batch,n,n,c=X.shape

        nn=int(np.floor((n-self.size)/self.stride))+1

        Xn=np.zeros((batch,nn,nn,c))

        for x in range(nn) :
            for y in range(nn) :
                Xn[:,x,y,:]=self.fnct(X[:,x*self.stride:x*self.stride+self.size,
                                y*self.stride:y*self.stride+self.size,:],axis=(1,2))
                
        return Xn.reshape(batch,-1,order='F')

    
class HOG(Extractor) : 
    def __init__(self,size,bin_direction,mask='classic'):
        self.size=size
        self.bin_direction=bin_direction
        self.mask=mask

    def transform(self,X) :
        X=X.mean(axis=-1)
        batch,n,n=X.shape

        #replication padding
        X_padded = np.zeros((batch,n+2,n+2))
        X_padded[:,1:-1,1:-1] = X
        X_padded[:,0,1:-1]=X[:,0,:]
        X_padded[:,-1,1:-1]=X[:,-1,:]
        X_padded[:,1:-1,0]=X[:,:,0]
        X_padded[:,1:-1,-1]=X[:,:,-1]

        X_padded[:,0,0]=X[:,0,0]
        X_padded[:,-1,0]=X[:,-1,0]
        X_padded[:,0,-1]=X[:,0,-1]
        X_padded[:,-1,-1]=X[:,-1,-1]


        #compute gradient
        dx=np.zeros((batch,n,n))
        dy=np.zeros((batch,n,n))
        if self.mask == 'classic':
            dx=-X_padded[:,:-2,1:-1]+X_padded[:,2:,1:-1]
            dy=-X_padded[:,1:-1,:-2]+X_padded[:,1:-1,2:]
        elif self.mask == 'sobel' :
            dx=-X_padded[:,:-2,:-2]\
               -2*X_padded[:,:-2,1:-1]\
               -X_padded[:,:-2,2:]\
               +X_padded[:,2:,:-2]\
               +2*X_padded[:,2:,1:-1]\
               +X_padded[:,2:,2:]
            dy=-X_padded[:,:-2,:-2]\
               -2*X_padded[:,1:-1,:-2]\
               -X_padded[:,2:,:-2]\
               +X_padded[:,:-2,2:]\
               +2*X_padded[:,1:-1,2:]\
               +X_padded[:,2:,2:]

        #compute norm and angle of gradient
        d = np.sqrt(dx ** 2 + dy ** 2) # [batch,n-1,n-1]
        theta = np.arctan2(dy,dx)+np.pi # [batch,n-1,n-1] of angle between 0 and 2*pi

        nn=int(n/self.size)

        hog = np.zeros((batch,nn,nn,self.bin_direction))

        for x in range(nn) :
            for y in range(nn) :
                dxy=d[:,x*self.size:(x+1)*self.size,y*self.size:(y+1)*self.size] #[batch,size,size]
                dxy=dxy.reshape((batch,-1),order='F') #[batch,size*size]
                thetaxy=theta[:,x*self.size:(x+1)*self.size,y*self.size:(y+1)*self.size] # [batch,size,size]
                thetaxy=thetaxy.reshape((batch,-1),order='F')-1e-10 #[batch,size,*size]
                direction=(self.bin_direction*thetaxy/(2*np.pi)).astype(int) #[batch,size*size] of int in [0,bin_direction-1]
                for i in range(self.bin_direction) :
                    hog[:,x,y,i]=(dxy*(direction==i)).sum(axis=-1)
        #normalisation
        hog=hog/np.sqrt((hog**2).sum(axis=-1)+1e-5)[:,:,:,None]
        return hog.reshape((batch,-1),order='F')
    
class SIFT(Extractor) :
    def __init__(self,nfeatures=10) :
        self.nfeatures=nfeatures
        self.sift=cv.SIFT_create(nfeatures=nfeatures,
                                contrastThreshold=1e-10)

    def transform(self, X):
        X=X.mean(axis=-1)
        X-=X.min()
        X/=X.max()
        X*=256-1e-5
        X=X.astype(np.uint8)

        X_transformed=np.zeros((len(X),128*self.nfeatures))
        for i in range(len(X)) :
            _,descriptor=self.sift.detectAndCompute(X[i],None)
            if descriptor is not None :
                descriptor=descriptor.flatten()
                n=min(128*self.nfeatures,len(descriptor))
                X_transformed[i,:n]=descriptor[:n]

        return X_transformed