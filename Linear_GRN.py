#ABCD GRN (Linear Gene Regulatory Network)

import numpy as np
import pandas as pd
from scipy.integrate import odeint

y0 = np.array([70,0,0,0,0])

#you can play around with these parameters

fA =10
bA= -15
fB =20
bB =-45
fC =30
bC =-15
fD =40
bD =-10


def dX_dt(X, t=0):
    return np.array([
        0,
        (fA*X[0]) +   (bA*X[1]) ,
        (fB*X[1]) + (bB*X[2]),
        (fC*X[2]) + (bC*X[3]),
        (fD*X[3]) + (bD*X[4])
    ])


t  = np.linspace(0,15,1000)

Xt = odeint(dX_dt, y0, t)

t_= np.reshape(t,(1000,1))
print(t_)

X_t=np.concatenate((t_,Xt),axis = 1)

df = pd.DataFrame(X_t)
df_=df.to_csv('simulated_ABCD.csv')

