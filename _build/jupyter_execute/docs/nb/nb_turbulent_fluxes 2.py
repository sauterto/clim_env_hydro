#!/usr/bin/env python
# coding: utf-8

# (nb_seb)=
# # Turbulent fluxes and surface energy balance
# 
# 
# 

# <div class="alert alert-block alert-success">
# <b>Learning objectives:</b><br>
# <ul>
#  <li>Parametrization of turbulent fluxes</li>
#  <li>Surface energy balance</li>
# </ul>  
# </div>

# <div class="alert alert-block alert-info">
# <b>Prerequisites</b><br>
# <ul>
# <li>Basic knowledge of Python, Jupyter Notebooks, and data analysis</li>
# <li>Familiarity with Scipy, MetPy, Pandas, Xarray, and Plotly</li>
# </ul>  
# </div>

# In[ ]:


import numpy as np
import pandas as pd
import math
import xarray as xr
from scipy.optimize import minimize

# Import the plotly library
from plotly.subplots import make_subplots
import plotly.subplots as sp
import plotly.graph_objs as go


# This exercise uses functions from the module HU_learning_material. The module was developed for this course and consists of a collection of functions discussed in the course. The module can be used with

# In[ ]:


# Import the learning module
from hu_tools import HU_learning_material


# This is a so-called Python class. The class is called **HU_learning_material** and must be instantiated. This is done with

# In[ ]:


#-----------------------------
# Load the learning materials
#-----------------------------"
hu = HU_learning_material()


# Eddy covariance (EC) stations measure turbulent flows directly with an ultrasonic anemometer and open path gas analysers. The sensors work fast and allow measurement frequencies of 25 Hz and more. EC stations are very expensive, so most weather stations are only equipped with low-frequency sensors. Standard weather stations measure temperature and relative humidity at several measurement levels. Average turbulent fluxes can be parameterised with the bulk approach by using the temperature and humidity measurements. The bulk approach depends on the temperature gradient, the humidity gradient, the wind velocity and the roughness length. Before we can parameterise the turbulent flows, we need to define some parameters

# In[ ]:


# Define simulation parameters
rho = 1.2        # Air density [kg m^-3]
z = 2.0          # Measurement height [m]
z0 = 1e-3        # Aerodynamic roughness length [m]
albedo = 0.5     # Albedo [-]


# Furthermore, we need the measured atmospheric quantities
# 

# In[ ]:


# Radiation, Wind velocity and pressure
G = 700.0        # Incoming shortwave radiation [W m^-2] 
U = 2.0          # Wind speed [m s^-1]
p = 700.0        # Surface air pressure [hPa]

# Temperature measurements
T0 = 310         # Temperature close to the surface [K]
T2 = 285         # Temperature at height z [K]

# Relative humidity measurements
f0 = 0.78        # Relative humidity close to the surface [%]
f2 = 0.8         # Relative humidity at height z [%]


# Constants
L = 2.83e6       # latent heat for sublimation
cp = 1004.0      # specific heat [J kg^-1 K^-1]


# For the calculation of the turbulent latent heat flow, the relative humidity must still be converted into the mixing ratio. This conversion can be done with the function **mixing_ratio()** which requires relative humidity, temperature, and pressure 

# In[ ]:


# Mixing ratio at level z (2 m) and near the surface
q2 = hu.mixing_ratio(f2,T2,p)
q0 = hu.mixing_ratio(f0,T0,p)


# <div class="alert alert-block alert-warning">
# <b>Broaden Knowledge & Comprehension</b></br>
# Use Metpy to calculate the mixing ratio
# </div>

# The bulk coefficients $C_h$ and $C_e$ for the sensible and latent heat can be derived with the function **bulk_coeff_shf()** and **bulk_coeff_lhf()**. The functions require the measurement heigth *z* and the roughness length *$z_0$*.

# In[ ]:


# Bulk coefficient for latent heat
Ce = hu.bulk_coeff_lhf(z, z0)
# Bulk coefficient for sensible heat
Ch = hu.bulk_coeff_shf(z, z0)


# Finally, the turbulent fluxes can then be calculated

# In[ ]:


# Sensible heat flux
lhf = hu.lhf_bulk(rho, L,  Ce, U, q2, q0)
# Latent heat flux
shf = hu.shf_bulk(rho, cp, Ch, U, T2, T0)

print('SHF: {:.2f}'.format(shf))
print('LHF: {:.2f}'.format(lhf))


# The energy balance of surfaces assumes that the sum of all energy fluxes is zero, i.e. Q+H+L+B=0 with Q the radiation balance, H the turbulent flux of sensible heat, L the latent heat flux and B the ground heat flux. The quantity via which the equation can be closed is the surface temperature. Strictly speaking, we are looking for the surface temperature at which this equation is fulfilled. This temperature can be found with the function **get_surface_temp()**.

# In[ ]:


# Optimize for surface temperature neglecting the ground heat flux
T_0 = hu.get_surface_temp(T2,f0,f2,albedo,G,p,rho,U,z,z0,B=0.0)
print('T0: {:.2f} K'.format(T_0[0]))


# In this example it was assumed that the soil heat flux does not play a role (B=0). We can now calculate the sensible heat flux based on the surface temperature

# In[ ]:


# Get the melt and sublimation rates
Q_0, H_0, E_0, LWd, LWu, SWu = hu.EB_fluxes(T_0,T2,f0,f2,albedo,G,p,rho,U,z,z0)

print('T0: {:.2f}'.format(T_0[0]))
print('G (incoming shortwave): {:.2f}'.format(SWu))
print('SW up(outgoing shortwave): {:.2f}'.format(G))
print('LW up (outgoing longwave): {:.2f}'.format(LWu[0]))
print('LW down (incoming longwave): {:.2f}'.format(LWd))
print('SHF (sensible heat flux): {:.2f}'.format(H_0[0]))
print('LHF (latent heat flux): {:.2f}'.format(E_0[0]))
print('Net radiation: {:.2f}'.format(Q_0[0]))


# <div class="alert alert-block alert-warning">
# <b>Broaden Knowledge & Comprehension</b></br>
# <ul>
# <li>Check if the energy balance is closed
# <li>If the equation is not closed, what could be the reasons? If it does, is the equation always closed?
# <li>In the current version, the soil heat flux can only be specified as a constant. What would have to be done to calculate the soil heat flux? Describe in keywords how you would proceed.
# <li>Discuss the differences between the Bulk approach and the Bowen-Ratio method, such as the measurements required, dependencies of parameters, difficulties and accuracy.
# <li>Calculate the Bowen ratio. What does this ratio say? What kind of ratio would you expect in mountainous regions?
# <li>Use a Bowen ratio of 0.3 and the given quantities from the previous tasks. Calculate the turbulent flux of sensible and latent heat 
# <li>An automatic weather station measures a net radiation of 623 $W m^{−2}$ and a ground flux of 64 $W m^{−2}$ over grassland. Find the other terms of the surface heat budget using the Bowen-ratio method. (Hint: Assume a realistic Bowen-ratio)
# </ul>
# </div>

# We can now examine the sensitivity of the fluxes. For example, we can easily investigate how the wind velocity affects the surface temperature and the turbulent sensible heat flux.

# In[ ]:


# Define result arrays
Ua = []  # stores the wind velocities
Ta = []  # stores the surface temperatures
Ha = []  # stores the sensible heat flux
Ea = []  # stores the latent heat flux

# Do a loop over a range of wind velocities, e.g. from 0 to 10 m/s
for U in np.arange(0,10,0.1):
    # Append the wind velocity to the array
    Ua.append(U)
    
    # Calculate the new surface temperature using the new wind velocity
    T0 = hu.get_surface_temp(T2,f0,f2,albedo,G,p,rho,U,z,z0,B=0.0)
    
    # Store the new surface temperature to the array
    Ta.append(T0[0])
    
    # Get the fluxes
    Q,H,E,LWd,LWu,SWu = hu.EB_fluxes(T0,T2,f0,f2,albedo,G,p,rho,U,z,z0)
    
    # Append the fluxes to the arrays
    Ha.append(H[0])
    Ea.append(E[0])


# We can now plot the results

# In[ ]:


#----------------------
# Plot the results
#----------------------
# Creating the plot with two rows and one column. The plots share the same x-axis so that only labels 
# for the lower plots are shown
fig = make_subplots(rows=2, cols=1, shared_xaxes=True)

# Adding the transmissivity data as a  line in the top panel
fig.add_trace(go.Scatter(x=Ua, y=Ta, line=dict(color='royalblue', dash='solid'), name='Surface temperature [K]'),
                        row=1, col=1)

# Adding the local elevation angle as line in the bottom panel
fig.add_trace(go.Scatter(x=Ua, y=Ha, line=dict(color='green', dash='solid'), name='Sensible heat flux [W m^-2]'),
                         row=2, col=1)

# Adjusting the layout
fig.update_layout(title='Dependency of the surface temperature and heat flux on wind velocity', plot_bgcolor='white', width=800, height=600,
                  yaxis=dict(showgrid=True, gridcolor='lightgray', gridwidth=1),
                  yaxis2=dict(showgrid=True, gridcolor='lightgray', gridwidth=1),
                  xaxis=dict(title='', showgrid=True, gridcolor='lightgray', gridwidth=1),
                  xaxis2=dict(title='Wind velocity',  showgrid=True, gridcolor='lightgray', gridwidth=1))


# Adjusting the axes
fig.update_xaxes(nticks=10, row=1, col=1)
fig.update_yaxes(nticks=5, row=1, col=1)
fig.update_xaxes(nticks=10, row=2, col=1)
fig.update_yaxes(nticks=5, row=2, col=1)

# Showing the plot
fig.show()


# <div class="alert alert-block alert-warning">
# <b>Broaden Knowledge & Comprehension</b></br>
# <ul>
# <li>Why are the curves not smooth but show noticeable jumps?
# <li>Explain why the surface temperature decreases with increasing wind speed while the turbulent flow of sensible heat increases at the same time.
# <li>Conduct your own experiments, for example by varying the relative humidity or the roughness length. Look closely at how the fluxes change.
# </ul>
# </div>

# <div class="alert alert-block alert-success">
# <b>Evaluate and Analyze</b></br>
# <ul>
# <li>The turbulent fluxes can also be determined using the so-called K-approach with $$Q=\rho \cdot c_p \cdot \overline{w'T'}=\rho \cdot c_p \cdot K \cdot \frac{\Delta T}{\Delta z},$$ where K is the turbulent diffusivity in m^2 s^{-1}. For the latent heat flow, the mixing ratio is also used instead of the temperature. Write a function to calculate the turbulent fluxes with the K-approach. Use a value of 5 m^2 s^{-1} for K and select suitable values for the other variables. Compare the results with those of the bulk approach. Choose suitable values for K so that you get the same result.
# <li>Extend the model by using the common parameterisation $$K=\kappa \cdot z \cdot u_{*}$$ with $$\kappa=0.41$$ the von Karman constant.
# <li>Write a function for the logarithmic wind profile in the form <b>windprofile(z, z0, ustar)</b>. Perform sensitivity tests and analyse how the wind profile changes when you change z0 and ustar.
# <li>On an overcast day, a wind speed of 6 m/s is measured with an anemometer located 10 m over ground within a corn field. What is the wind speed at 25 m height?
# <li>A wind speed of 3.15 m/s is measured at 10 m heights. Find the friction velocity using an analytical approach (logarithmic wind law).
# </ul>
# </div>

# In[ ]:




