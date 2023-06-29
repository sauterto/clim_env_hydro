#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np

# Load CSV file
df = pd.read_csv("https://raw.githubusercontent.com/sauterto/clim_env_hydro/main/docs/nb/data/FLX_CH-Dav.csv", parse_dates=True, index_col=0)

# The file contains:
# Air temperature       :: t2m
# Relative humdity      :: RH
# Precipitation         :: precip
# Wind speed            :: WS
# Wind direction        :: WD
# Net radiation         :: NETRAD
# Incoming shortwave    :: SW_IN
# Outgoing shortwave    :: SW_OUT
# Incoming longwave     :: LW_IN
# Outgoing longwave     :: LW_OUT
# Sensible heat flux    :: H
# Latent heat flux      :: LE
# Ground heat flux      :: QG
# Surface temperature   :: TS


# In[17]:


df.hist(column='WS', bins=np.arange(0,20,0.1));


# In[19]:


import scipy.stats as stats
A = stats.weibull_min(df.WS, )


# In[23]:


A


# In[3]:


# Import rosely
from rosely import WindRose

# Create a WindRose object with the resampled hourly data
WR = WindRose(resampled_hourly)

# create renaming dictionary - the WindRose object requires the variable names ws and wd. We have
# to tell the Object the name of our Variables: WS, WD
names = {'WS':'ws', 'WD':'wd'}

# calculate wind statistics for 8 sectors
WR.calc_stats(normed=False, bins=8, variable_names=names)

# Generate windrose plot
WR.plot(
    template='plotly_dark',
    colors='haline',
    title='Davos, Switzerland',
    output_type='show',
    width=600,
    height=600
)

# To view the results of the wind statistics that will be used for the wind rose later, 
# view the WindRose.wind_df which is created after running WindRose.calc_stats()
# Here we all speed and fequencies for the northerly sector
WR.wind_df.loc[WR.wind_df.direction=='N']


# In[ ]:




