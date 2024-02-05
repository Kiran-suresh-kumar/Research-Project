#numerical_simulations_prey_predator
#data_set_creation

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
from numpy import linalg as LA
c = sns.color_palette("Blues",20)

#you can play around with these terms 
e1 = 1.
g1 = -0.1
e2 = -1.5
g2 = 0.1

#defining the ODE, one wants to solve

def dX_dt(X, t=0):
    return np.array([
        e1*X[0] +   g1*X[0]*X[1] ,
        e2*X[1] + g2*X[0]*X[1]
    ])


y0 = np.array([10, 5])
t  = np.linspace(0,15,1000)

Xt = odeint(dX_dt, y0, t)


ey, predator, data_time=[], [], []

prey=Xt[:,0]
predator=Xt[:,1]
data_time=t


#plotting the data
pylab.rcParams['figure.figsize'] = (16, 4.)
plt.plot(data_time,prey,'-d',color='blue', label='prey')
plt.plot(data_time,predator,'-d',color='orange', label='predator')
plt.legend()
plt.show()

#saving the data to CSV file
t_= np.reshape(t,(1000,1))
X_t=np.concatenate((t_,Xt),axis = 1)
df = pd.DataFrame(X_t)
df_=df.to_csv('simulated_prey_predator.csv')

