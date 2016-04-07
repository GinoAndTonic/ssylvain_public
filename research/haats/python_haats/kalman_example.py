__author__ = 'ssylvain'
import numpy as np
from kalman import *
from scipy.linalg import expm
import matplotlib.pyplot as plt
import matplotlib as mplt
import pylab as plab
#%matplotlib inline

#Let's use 2 mean reverting state variables and 3 observed variables
n,m,T=2,6,100
u=np.random.rand(n,n)
# K=u*diag(rand(n,1))*u';eig(K); %this we impose eigvals with real positive components
K=np.mat(np.diag(np.random.rand(n)))
#this we impose eigvals with real positive components
theta=np.random.rand(n,1) #long run mean for x.
y=np.mat(np.ones((T,m))*np.nan)
x=np.mat(np.ones((T,n))*np.nan)
x[0,:]=np.random.randn(1,n) #initial x
U1=np.mat(expm(-K*1/12)) #divide by 12 for monthly frequency
U0=(np.identity(n)-U1)*theta
A0=np.mat(np.random.rand(m,1)*10)
A1= np.mat(np.random.rand(m,n)*10)
Q=np.mat(np.identity(n)*0.2)
Phi=np.mat(np.identity(m)*1)
for t in range(T):
    if t > 0:
        x[t,:]=(U0 + U1*x[t-1,:].T + Q*np.random.randn(n,1)).T
    y[t,:] = (A0 + A1*x[t,:].T + Phi*np.random.randn(m,1)).T
plt.figure()
plt.plot(y)
plt.title('y')
plt.show()
plt.figure()
plt.plot(x)
plt.title('x')
plt.show()

kobj = Kalman(y, A0, A1, U0, U1, Q, Phi)
Ytt, Yttl, Xtt, Xttl, Vtt, Vttl, Gain_t,eta_t, cum_log_likelihood=kobj.filter()

for i in range(n):
    plt.figure()
    plt.plot(np.hstack((Xttl[:,i], x[:,i])))
    plt.legend({'filtered','actual'})
    plt.title('x'+str(i))
    plt.show()

print('stop here')