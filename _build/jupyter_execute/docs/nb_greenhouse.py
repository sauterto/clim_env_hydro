#!/usr/bin/env python
# coding: utf-8

# (greenhouse:exercise)=
# # Greenhouse model

# **Task 1**: Plug Eq. (7) into Eq. (6) and solve for the radiative equilibrium suface temperature $T_e$ 

# In[1]:


# Solve for the radiative equilibrium temperature Te
# Put you own code here
sigma = 5.67e-8
Q = 342
albedo = 0.3

Te = (((1-albedo)*Q)/sigma)**(1/4)

print('Radiative equilibrium temperature: {:.2f}'.format(Te))


# **Task 2**: Where in the atmosphere do we find $T_e$?

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import xarray as xr


# In[3]:


## NCAR data
url = 'air.mon.ltm.1981-2010.nc'
nc = xr.open_dataset('./files/air.mon.ltm.1981-2010.nc', use_cftime = True)
nc


# In[4]:


#  Take global, annual average and convert to Kelvin
nc.air.sel(time='0001-06-01 00:00:00',level=1000).plot(figsize=(20,10))


# In[5]:


#  a "quick and dirty" visualization of the data
weights = np.cos(np.deg2rad(nc.lat))
air_weighted = nc.air.weighted(weights)

weighted_mean = air_weighted.mean(("lat",'lon','time'))
weighted_mean.plot()


# In[ ]:





# In[13]:


# Create the skew-plot with the metpy module
from metpy.plots import SkewT

fig = plt.figure(figsize=(10,7))

skew = SkewT(fig, rotation=30)
skew.plot(weighted_mean.level, weighted_mean, color='black', linestyle='-')

skew.plot_dry_adiabats(linewidth=0.5)
skew.plot_moist_adiabats(linewidth=0.5)


# **Task 3**: What is the surface temperature with the single layer model? 

# In[14]:


# Solve for the atmospheric surface temperature

# Calc surface temperature Ts
# Put you own code here
Ts = 2**(1/4) * Te

print('Surface temperature: {:.2f}'.format(Ts))


# Why does the model overestimate the surface temperature?

# **Task 5**: Write a Python function for $OLR = U_2 = (1-\epsilon)^2 \sigma T_s^4 + \epsilon(1-\epsilon)\sigma T_0^4 + \epsilon \sigma T_1^4$

# In[17]:


def two_layer_model(Ts, T0, T1, epsilon):
    return ((1-epsilon)**2)*sigma*Ts**4 + epsilon*(1-epsilon)*sigma*T0**4 \
+ epsilon*sigma*T1**4


# **Task 7**: We will tune our model so that it reproduces the observed global mean OLR given observed global mean temperatures. Determine the temperatures for the two-layer model from the following sounding

# ![alt text](pics/vertical_profile.png "Sounding")

# **Task 8**: Find graphically the best fit value of $\epsilon$
# 

# In[20]:


import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

# Assignments
OLR = []        # initialize array
epsilons = []   # initialize array
OLR_obs = 238.5 # observed outgoing long-wave radiation

# Optimize epsilon
# We define a range from 0 to 1 with a 0.01 step and calculate the OLR for each of these epsilon values
for eps in np.arange(0, 1, 0.01):
    OLR.append(OLR_obs - two_layer_model(288, 275, 230, eps))
    # Store the results in the epsilon-array
    epsilons.append(eps)   
    
def find_nearest(array, value):
    """ 
    Auxiliary function to find index of closest value in an array 
    array :: input array
    value :: the value to find in the array
    """
    # This searches the minimum between the values and all values in an array
    # Basically, we enumerate over the array. The enumerator iterates over the array and returns a 
    # tuple (index, value) for each element. We take the value (x[1]) from this tuple and substract the value
    # we are searching for. The element which has the smallest difference is what we are looking for. We finally
    # return the index of this value.
    idx,val = min(enumerate(array), key=lambda x: abs(x[1]-value))
    return idx

# Now, find the closest value to the observed OLR using the previously defined function
idx = find_nearest(OLR, 0)

# Save the optimized epsilon in the epsilons-array
epsilon = epsilons[idx]

# Plot the results
print('The optimized transmissivity is: {:.2f}'.format(epsilons[idx]))
plt.figure(figsize=(15,8))
plt.plot(epsilons,OLR);
plt.scatter(epsilons[idx], OLR[idx], s=50,  color='r')
plt.hlines(0,0,1,linestyles='dotted',color='gray');
    


# In[21]:


two_layer_model(288, 275, 230, 0.59)


# **Task 9**: Write a Python function to calculate each term in the OLR. Plug-in the observed temperatures and the tuned value for epsilon.

# In[22]:


def two_layer_terms(Ts, T1, T0, epsilon):
    return ( ((1-epsilon)**2)*sigma*Ts**4, \
            epsilon*(1-epsilon)*sigma*T0**4, \
            epsilon*sigma*T1**4)


# In[25]:


# Calculate terms
term1, term2, term3 = two_layer_terms(288, 275, 230, 0.59)

print('Term Ts: {:.2f}'.format(term1))
print('Term T0: {:.2f}'.format(term2))
print('Term T1: {:.2f}'.format(term3))


# **Task 10**: Changing the level of emission by adding absorbers, e.g. by 10 %. 
# Suppose further that this increase happens abruptly so that there is no time for the temperatures to respond to this change. We hold the temperatures fixed in the column and ask how the radiative fluxes change.
# 
# Which terms in the OLR go up and which go down?

# In[14]:


# Make simulation here


# **Task 11**: Calculate the radiative forcing for the previous simulation

# In[15]:


# Calculate radiative forcing


# **Task 12**: What is the greenhouse effect for an isothermal atmosphere?

# In[16]:


# Make simulation here


# **Task 13**: For a more realistic example of radiative forcing due to an increase in greenhouse absorbers, we use our observed temperatures and the tuned value for epsilon. Assume an increase of epsilon by 2 %.

# In[17]:


# Make simulation here

