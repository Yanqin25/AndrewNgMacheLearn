import numpy as np
from sigmoid import sigmoid
def predict(Theta1,Theta2,X):
    m=X.shape[0]
    X=np.c_[np.ones(m),X]
    # 100*25
    a1=sigmoid(X.dot(Theta1.T))
    a1=np.c_[np.ones(a1.shape[0]),a1]
    # 100*10
    pred=sigmoid(a1.dot(Theta2.T))
    return np.argmax(pred,axis=1)+1