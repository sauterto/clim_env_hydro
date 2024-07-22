#!/usr/bin/env python
# coding: utf-8

# In[1]:


import xarray as xr
import metpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


# Load  dataset
ds = xr.open_dataset('./era5_exercise.nc')


# In[3]:


# Dump netcdf information
ds


# In[4]:


# Parse full dataset
data_parsed = ds.metpy.parse_cf()
data_parsed


# In[5]:


# Show CRS information
data_parsed['ro'].metpy.pyproj_crs


# In[6]:


# Convert precip data from meter to mm
precip = data_parsed['tp']*1000
precip.sel(time='2015').sum('time').plot()


# In[7]:


# Get the length of each monthly data record
month_length = ds.time.dt.days_in_month

# Plot the result
month_length


# In[8]:


# Calculate the weights by grouping by 'time.season'.
weights = (
    month_length.groupby("time.season") / month_length.groupby("time.season").sum()
)

# Test that the sum of the weights for each season is 1.0
np.testing.assert_allclose(weights.groupby("time.season").sum().values, np.ones(4))


# In[9]:


# Calculate the weighted average
ds_weighted = (ds * weights).groupby("time.season").mean(dim="time")
ds_weighted


# In[10]:


ds_weighted['P-E'] = (ds_weighted['tp']-ds_weighted['e'])*1000


# In[11]:


# Quick plot to show the results
notnull = pd.notnull(ds_weighted["ro"][0])

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))

ds_weighted["P-E"].sel(season='DJF').where(notnull).plot.pcolormesh(
    ax=axes[0, 0],
    cmap="Spectral_r",
    add_colorbar=True,
    extend="both",
    vmin=0.0, vmax=0.5,
)
ds_weighted["P-E"].sel(season='MAM').where(notnull).plot.pcolormesh(
    ax=axes[0, 1],
    cmap="Spectral_r",
    add_colorbar=True,
    extend="both",
    vmin=0.0, vmax=0.5,
)
ds_weighted["P-E"].sel(season='JJA').where(notnull).plot.pcolormesh(
    ax=axes[1, 0],
    cmap="Spectral_r",
    add_colorbar=True,
    extend="both",
    vmin=0.0, vmax=0.5,
)
ds_weighted["P-E"].sel(season='SON').where(notnull).plot.pcolormesh(
    ax=axes[1, 1],
    cmap="Spectral_r",
    add_colorbar=True,
    extend="both",
    vmin=0.0, vmax=0.5,
)

for ax in axes.flat:
    ax.axes.get_xaxis().set_ticklabels([])
    ax.axes.get_yaxis().set_ticklabels([])
    ax.axes.axis("tight")
    ax.set_xlabel("")

plt.tight_layout()

fig.suptitle("Seasonal Surface Air Temperature", fontsize=16, y=1.02)


# In[ ]:




