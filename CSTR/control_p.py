import numpy as np
import scipy as sc
#from Simul import CSTRsys

class controller():
    #function (model,t,Xs)
    def __init__(self,sys):
        self.sys=sys

    def ss_lqr_gain(self,A,B,Q,R):
        A=np.array(A).astype('float')
        B=np.array(B).astype('float')
        X=np.matrix(sc.linalg.solve_continuous_are(
            A,
            B,
            Q,
            R))
        self.K=-1*np.matrix(sc.linalg.inv(R)*(np.matrix(B.T)*X))
        return self.K
    
    def u_gain_based(self,Xs):
        u=np.matmul(self.K,Xs)
        return np.array(u)

    #def linear_MPC(A,B,Q,R):


