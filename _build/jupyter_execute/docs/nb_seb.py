#!/usr/bin/env python
# coding: utf-8

# (seb:exercise)=
# ### Surface Energy Balance model
# 
# The aim of this exercise is to understand how we can solve simple optimisation problems. To do this, we will develop a simple surface energy balance model (SEB). Since energy can neither be produced nor destroyed, the sum of the energy fluxes at the Earth's surface must be zero. If the static quantities such as roughness length, albedo, stability etc. are known and quantities such as temperature and humidity are measured, the balance of the energy fluxes at the surface is only a function of the surface temperature.
# 
# For simplicity, we parameterise the turbulent fluxes with a bulk approach and neglect the soil heat flux. However, at the end of this exercise, we will consider the soil heat flux by coupling the heat conduction equation to the energy balance model.

# **Task 1**: Develop a simple SEB model. The turbulent flows are to be parameterised using a simple bulk approach. Write a function that takes the following arguments: surface temperature, air temperature, relative humidity, albedo, global radiation, atmospheric pressure, air density, wind speed, altitude measured and roughness length. The function should return the short-wave radiation balance and the two turbulent energy fluxes.

# In[ ]:


import math
import numpy as np
from scipy.optimize import minimize, minimize_scalar
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


def EB_fluxes(T_0,T_a,f,albedo,G,p,rho,U_L,z,z_0):
    """ This function calculates the energy fluxes from the following quantities:
    
    Input: 
    T_0       : Surface temperature, which is optimized [K]
    f         : Relative humdity as fraction, e.g. 0.7 [-]
    albedo    : Snow albedo [-]
    G         : Shortwave radiation [W m^-2]
    p         : Air pressure [hPa]
    rho       : Air denisty [kg m^-3]
    z         : Measurement height [m]
    z_0       : Roughness length [m]
    
    """
    
    # Some constants
    c_p = XXXXXX            # specific heat [J kg^-1 K^-1]
    kappa = XXXXXX          # Von Karman constant [-]
    sigma = XXXXXX       # Stefan-Bolzmann constant
    
    # Bulk coefficients 
    Cs_t = XXXXXX 
    Cs_q = XXXXXX  
    
    # Correction factor for incoming longwave radiation
    eps_cs = 0.23 + 0.433 * np.power(100*(f*E_sat(T_a))/T_a,1.0/8.0)
    
    # Select the appropriate latent heat constant
    L = 2.83e6 # latent heat for sublimation

    # Calculate turbulent fluxes
    H_0 = XXXXXX
    E_0 = XXXXXX
    
    # Calculate radiation budget
    L_d = XXXXXX
    L_u = XXXXXX
    Q_0 = XXXXXX

    return (Q_0, L_d, L_u, H_0, E_0)

def E_sat(T):
    """ Saturation water vapor equation """
    Ew = 6.112 * np.exp((17.67*(T-273.16)) / ((T-29.66)))
    return Ew


# In[ ]:


# Test the SEB function
# Define necessary variables and parameters
T_0 = 283.0   # Surface temperature
T_a = 280.0   # Air temperature 
f = 0.7       # Relative humidity
albedo = 0.3  # albedo
G = 700.0     # Incoming shortwave radiation
rho = 1.1     # Air density
U = 2.0       # Wind velocity
z =  2.0      # Measurement height
z0 = 1e-2     # Roughness length
p = 1013      # Pressure

# Run the function
Q_0, L_d, L_u, H_0, E_0 = EB_fluxes(T_0,T_a,f,albedo,G,p,rho,U,z,z0)

# Print results
print('Surface temperature: {:.2f}'.format(T_0))
print('Global radiation: {:.2f}'.format(Q_0))
print('Longwave down: {:.2f}'.format(L_d))
print('Longwave up: {:.2f}'.format(L_u))
print('Surface heat flux: {:.2f}'.format(H_0))
print('Latent heat flux: {:.2f}'.format(E_0))
print('Energy Balance: {:.2f}'.format(Q_0+L_d-L_u-H_0-E_0))


# **Task 2**: Now we need to optimize for the surface temperature. Therefore, we need to write a so-called optimization function. In our case the sum of all fluxes should be zero. The SEB depends on the surface temperature. So we have to find the surface temperature which fulfills the condition $SEB(T_0)=Q_0+L_d-L_u-H_0-E_0=0$. 

# In[ ]:


def optim_T0(x,T_a,f,albedo,G,p,rho,U_L,z,z0):
    """ Optimization function for surface temperature:
    
    Input: 
    T_0       : Surface temperature, which is optimized [K]
    f         : Relative humdity as fraction, e.g. 0.7 [-]
    albedo    : Snow albedo [-]
    G         : Shortwave radiation [W m^-2]
    p         : Air pressure [hPa]
    rho       : Air denisty [kg m^-3]
    z         : Measurement height [m]
    z_0       : Roughness length [m]
    
    """
    
    Q_0, L_d, L_u, H_0, E_0 = EB_fluxes(x,T_a,f,albedo,G,p,rho,U_L,z,z0)
    
    # Get residual for optimization
    res = np.abs(Q_0+L_d-L_u-H_0-E_0)

    # return the residuals
    return res


# We use the **minimize function** from the scipy module to find the temperature values. 

# In[ ]:


optim_T0(293.5,T_a,f,albedo,G,p,rho,U,z,z0)


# In[ ]:


# Test the SEB function
# Define necessary variables and parameters
T_0 = 283.0   # Surface temperature
T_a = 280.0   # Air temperature 
f = 0.7       # Relative humidity
albedo = 0.3  # albedo
G = 700.0     # Incoming shortwave radiation
rho = 1.1     # Air density
U = 2.0       # Wind velocity
z =  2.0      # Measurement height
z0 = 1e-3     # Roughness length
p = 1013      # Pressure

# Run the function
res = minimize(optim_T0,x0=XXXXXX,args=(XXXXXX),bounds=((XXXXXX,XXXXXX),), \
                         method='L-BFGS-B',options={'eps':1e-8})

print('Result: {:} \n'.format(res))
print('Optimizes T0: {:.2f}'.format(res.x[0]))


# The temperature value is stored in the x value of the result dictionary

# In[ ]:


# Assign optimization result to variable T_0
T_0 = XXXXXX

# Run the function
Q_0, L_d, L_u, H_0, E_0 = EB_fluxes(T_0,T_a,f,albedo,G,p,rho,U,z,z0)

# Print results
print('Surface temperature: {:.2f}'.format(T_0))
print('Global radiation: {:.2f}'.format(Q_0))
print('Longwave down: {:.2f}'.format(L_d))
print('Longwave up: {:.2f}'.format(L_u))
print('Surface heat flux: {:.2f}'.format(H_0))
print('Latent heat flux: {:.2f}'.format(E_0))
print('Energy Balance: {:.2f}'.format(Q_0+L_d-L_u-H_0-E_0))


# In[ ]:




