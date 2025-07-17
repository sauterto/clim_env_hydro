#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import pandas as pd
import math
import xarray as xr
from scipy.optimize import minimize

# Import the plotly library
from plotly.subplots import make_subplots
import plotly.subplots as sp
import plotly.graph_objs as go

# Import the learning module
from hu_tools import HU_learning_material

#-----------------------------
# Load the learning materials
#-----------------------------"
hu = HU_learning_material()


# In[5]:


#-----------------------
# Define simulation parameters 
# @ANTONIA: Change this parameters 
#-----------------------
rho = 1.2        # Air density [kg m^-3]
z = 2.0          # Measurement height [m]
z0 = 1e-3        # Aerodynamic roughness length [m]
albedo = 0.5     # Albedo [-]


# Constants (do not change)
L = 2.83e6       # latent heat for sublimation
cp = 1004.0      # specific heat [J kg^-1 K^-1]

#-----------------------
# Boundary conditions (Heat Equation)
# @ ANTONIA: Change here is necessary
#-----------------------
Tb = 273.15    # Temperature at the bottom [K]
D  = 2.    # Depth [m]
Nz = 100   # Number of grid points in the vertical (soil)
dz = D/Nz
dt = 60.   # time step [s]
t  = 3600  # integration time [s]
alpha = 1.2e-6 # soil thermal diffusivity [m^2 s^-1]

#-----------------------------
# @ANTONIA: HERE you should read in the data
#-----------------------------
# Read file
df = pd.read_csv('antonia_test.csv',index_col='Date')
print(df)

#-----------------------
# Some more definition (do not change this)
#-----------------------
# Don't delete this
Tsoil = 273.0*np.ones(Nz)
result = np.zeros((Nz,len(df)))
B = 0.0
i = 0

#-----------------------
# @ANTONIA: Change the column names if necessary
#-----------------------
for date, row in df.iterrows():
    T_0 = hu.get_surface_temp(row['T_2'],row['f0'],row['f2'],row['albedo'],row['G'],row['p'],rho,row['U'],z,z0,B=B)
    Tsoil, dz = hu.heat_equation(T_0, Tb, D, Nz, t, dt, alpha, Tini=Tsoil)
    result[:,i] = Tsoil
    B = (Tsoil[1]-Tsoil[0])/dz
    i=i+1

# The result is stored in the variable result


# In[3]:


#-----------------------------
# Plot the results
#-----------------------------
# Generate the plotting grid
# y-axis values
y_values = np.arange(-D,0,dz)
# then the x-axis values
x_values = np.arange(len(df.index))
# Generate the 2D plotting grid
X, Y = np.meshgrid(x_values, y_values)

# Create the filled contour trace
trace = go.Contour(
    x=df.index,
    y=y_values,
    z=result[::-1],
    colorscale='Viridis',
    contours=dict(
        coloring='fill',
        showlabels=True
    ),
    colorbar=dict(
        title=dict(
            text='Temperature [ºC]',
            #standoff=15,
            side='right'
        )
    )
)

# Create the layout
layout = go.Layout(
    title='Soil temperature',
    xaxis=dict(title='Date', tickformat='%d.%m.%Y', tickangle=45, dtick='D3'),
    yaxis=dict(title='Depth [m]'),
    width=1000, height=500,
)

# Create the figure
fig = go.Figure(data=[trace], layout=layout)

# Show the plot
fig.show()


# In[ ]:




