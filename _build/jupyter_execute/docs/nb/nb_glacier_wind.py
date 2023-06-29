#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import matplotlib.pyplot as plt

Km = 0.1
Kh = 0.02
g = 9.81
lapse = 0.004
s = -5
s = np.sin(s*3.14/180)
C = -11
T0 = 280
zmax = 50

z_meas = [0.2,0.5,1,2,4,6,8,13]
u_meas = [3,3.2,3.8,4.2,4.3,4.4,4.3,3.9]
t_meas = [3.8,4.0,4.4,5.2,7.1,9,9.7,10]

lam = ((4*T0*Km*Kh)/(g*s**2*lapse))**(0.25)
mu = ((g*Kh)/(T0*Km*lapse))**(0.5)

theta1 = [C*np.exp(-z/lam)*np.cos(z/lam) for z in np.arange(0,zmax,0.5)]
u1 = [C*mu*np.exp(-z/lam)*np.sin(-z/lam) for z in np.arange(0,zmax,0.5)]

lam2 = ((g*s**2*lapse)/(4*T0*Km*Kh))**(0.25)
theta2 = [C*np.exp(-z*lam2)*np.cos(z*lam2) for z in np.arange(0,zmax,0.5)]
u2 = [C*np.exp(-z*lam2)*np.sin(z*lam2) for z in np.arange(0,zmax,0.5)]

#plt.plot(np.array(theta1)+11,np.arange(0,zmax,0.5))
#plt.plot(np.array(u1),np.arange(0,zmax,0.5))

#plt.plot(np.array(u_meas),np.array(z_meas))
#plt.plot(np.array(t_meas),np.array(z_meas))
plt.plot(np.array(theta2)/-C,np.arange(0,zmax,0.5))
plt.plot(np.array(u2)/(mu*C),np.arange(0,zmax,0.5))


# In[ ]:


np.arange(0,10,0.1)


# In[ ]:




