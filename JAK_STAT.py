#data generator for JAK-STAT pathway
%pylab inline
pylab.rcParams['figure.figsize'] = (12, 8)
import numpy as np
import math
from mpl_toolkits.mplot3d import Axes3D
#%matplotlib notebook
from Iterative_solvers_l0_l1 import *
import scipy.io as sio
import itertools
from operator import itemgetter, attrgetter
import matplotlib.pyplot as plt
from sklearn.linear_model import lars_path, enet_path, lasso_path
from itertools import cycle
from scipy.ndimage import gaussian_filter1d
from scipy.integrate import odeint
import pandas as pd
import numpy as np
import csv
from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression
from scipy.ndimage import gaussian_filter
import seaborn as sns
c = sns.color_palette("Blues",20)

#e1 (-k1) e2 (-k2) ...
#g1 (k1)  g2 (k2)...

#Reaction constants
e1 = -0.021
e2= -2.46
e3= -0.2066
e4=-0.10658

g1 = 0.021
g2 = 2.46
g3= 0.2066
g4= 0.10658

lambda1=0.07
lambda2=0.1

#defining the ODE-model
#rec is the model that can replicate c(t) in the paper (source: https://cran.r-project.org/web/packages/dMod/vignettes/dMod.html)
def dX_dt(X, t):
    rec= (((1 - np.exp(-t * lambda1)) * np.exp(-t * lambda2))**3)*10000
    print(t)
    print("rec:", rec)
    return np.array([
        (e1*X[0]*rec) +   2*(g4*X[3]) ,
        (e2*(X[1]*X[1])) + (g1*X[0]*rec),
        (e3*X[2])+(0.5*g2)*(X[1]*X[1]),
        (e4*X[3])+(g3*X[2])
    ])

#defining the initial conditions and the time interval 
y0 = np.array([6, 0, 0, 0])
t  = np.linspace(0,60,100)

#solving the ODE
Xt = odeint(dX_dt, y0, t)

t  = np.linspace(0,60,100)
lambda1=0.07
lambda2=0.1

#plotting the function (just to compare with the paper Maddu et al.
#the data is not exactly the same but similiar trend is seen\
# to get the exact trend better fit to the data is needed (Tikonov fit or fitting with regularizer)


rec= (((1 - np.exp(-t * lambda1)) * np.exp(-t * lambda2))**3)*10000
pylab.rcParams['figure.figsize'] = (16, 4.)
plt.plot(t,rec,'-d',color='blue', label='X1')

plt.legend()
plt.show()

data_time=[]

X1=Xt[:,0]
X2=Xt[:,1]
X3=Xt[:,2]
X4=Xt[:,3]
data_time=t


#plotting the data
pylab.rcParams['figure.figsize'] = (16, 4.)
plt.plot(data_time,X1,'-d',color='blue', label='X1')
plt.plot(data_time,X2,'-d',color='orange', label='X2')
plt.plot(data_time,X3,'-d',color='green', label='X3')
plt.plot(data_time,X4,'-d',color='red', label='X4')
plt.legend()
plt.show()

#converting into data CSV file

t_= np.reshape(t,(100,1))
X_t=np.concatenate((t_,Xt),axis = 1)
print(X_t.shape)
df = pd.DataFrame(X_t)
df_=df.to_csv('simulated_JAK_STAT.csv')
