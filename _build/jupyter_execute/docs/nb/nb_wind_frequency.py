#!/usr/bin/env python
# coding: utf-8

# (nb_wind_freq)=
# 
# # Wind frequency
# 
# Wind speeds are rarely constant. At any one location, wind speeds might be strong only rarely during a year, moderate many hours, light even more hours, and calm less frequently. The number of times than a range $\Delta U$ of wind speeds occured in the past is the frequency of occurrence. Dividing the frequency by the total number of wind observations gives a relative frequency. The expectation that this same relative frequency will occure in the future is the probability $Pr$. The probability distribution of mean wind speeds $U$ is described by the Weibull distribution. In this notebook we estimate the parameters of the Weibull distriubtion to fit the observations. From this we derive the return periods of wind events.

# <div class="alert alert-block alert-success">
# <b>Learning objectives:</b><br>
# <ul>
#  <li>Probability distribution of mean wind speeds</li>
#  <li>Return periods</li>
# </ul>  
# </div>

# <div class="alert alert-block alert-info">
# <b>Prerequisites</b><br>
# <ul>
# <li>Basic knowledge of Python, Jupyter Notebooks, and data analysis</li>
# <li>Familiarity with Scipy, Pandas, Xarray, and Plotly</li>
# </ul>  
# </div>

# In[ ]:


import numpy as np
import pandas as pd
import math
import xarray as xr
from rosely import WindRose

# Import the plotly library
from plotly.subplots import make_subplots
import plotly.subplots as sp
import plotly.graph_objs as go


# We use the measurement data from Davos station for the exercise

# In[ ]:


# Load CSV file
df = pd.read_csv("https://raw.githubusercontent.com/sauterto/clim_env_hydro/main/docs/nb/data/FLX_CH-Dav.csv", parse_dates=True, index_col=0)

#---------------------
# The file contains:
#---------------------
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

df


# We first plot the data

# In[ ]:


df['WS'].plot()


# Often we are interested in the distribution of the wind speed which can be plotted by a histogram

# In[ ]:


# Subplots
fig, ax = plt.subplots(1, 1);

# Plot the data histogram
ax.hist(df.WS, bins=100, density=True, histtype='stepfilled', alpha=0.8);


# Lets have also a look at the wind direction

# In[ ]:


# Create a WindRose object with the resampled hourly data
WR = WindRose(df)

# create renaming dictionary - the WindRose object requires the variable names ws and wd. We have
# to tell the Object the name of our Variables: WS, WD
names = {'WS':'ws', 'WD':'wd'}

# calculate wind statistics for 8 sectors
WR.calc_stats(normed=False, bins=8, variable_names=names)

# Generate windrose plot
WR.plot(
    template='xgridoff',
    colors='haline',
    title='Davos, Switzerland',
    output_type='show',
    width=600,
    height=600
)


# The probability distribution of mean wind speeds $U$ is described by the Weibull distribution:
# 
# $$
# Pr = \frac{\alpha \cdot \Delta U \cdot U^{\alpha-1}}{M_0^{\alpha}} \cdot exp \left[ - \left( \frac{M}{M_0} \right)^{\alpha}\right],
# $$
# 
# where Pr is the probability of wind speed $U \pm 0.5 \cdot \Delta U$. We can estimate the location parameter $M_0$ and the spread $\alpha$ with

# In[ ]:


# Fit the Weibull distribution to the wind speed data
shape, loc, scale = stats.weibull_min.fit(df.WS, loc=0, method="MLE")


# We can use the estimated parameters to get the probabilities for different wind speeds with 

# In[ ]:


# Get the Weibull distribution using the fitted parameters
rv = stats.weibull_min(c=shape, loc=loc, scale=scale)

# Value range for plotting the distribution
x = np.linspace(rv.ppf(0.01),
                rv.ppf(0.99999), 100)


# and plot the observations and the estimated Weibull distribution (red line)

# In[ ]:


# Subplots
fig, ax = plt.subplots(1, 1)

# Plot the data histogram
ax.hist(df.WS, bins=100, density=True, histtype='stepfilled', alpha=0.2)

# and the Weibull distribution
ax.plot(x, rv.pdf(x), 'r-', lw=2, alpha=0.6, label='Weibull pdf')


# One can express extreme-wind likelihood as a return period, which is equal to the total period of measurement divided by the number of times the wind exceeded a threshold. For example, we can estimat the number of observations lower than 12.5 m/s from the cumulative distribution

# In[ ]:


# Estimate from the cumulative distribution the number of observation lower than 12.5 m/s
cdfx = rv.cdf(12.5)
cdfx


# In our case, the data set includes 18 years. So we divide the number of years by the number of cases which are above this threshold, i.e. total number of elements minus the total observations multiplied by the percentage of all cases below the threshold. 

# In[ ]:


# Print max wind speed, and the return period
# Number of years in dataset
number_of_years = 18
print('Return period: {:.2f} years'.format(number_of_years/(len(df.WS)-len(df.WS)*cdfx)))


# This means that statistically a half-hour average wind speed of 12.5 m/s occurs only every 43.67 years.

# <div class="alert alert-block alert-warning">
# <b>Broaden Knowledge & Comprehension</b></br>
# <ul>
# <li>What is the mean wind speed? Check that the mean wind speed is proportional to the location parameter $M_0$
# <li>Assume that the distribution is obtained from a timeseries of length 100 year. How does the return period change? Why?
# </ul>
# </div>
