import matplotlib.pyplot as plt
import numpy as np

def imshow(X,**kwargs) :
    """
    Show a single image
    """
    X=X-X.min()
    X/=X.max()
    plt.imshow(X,**kwargs)

