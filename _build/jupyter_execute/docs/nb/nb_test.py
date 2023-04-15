#!/usr/bin/env python
# coding: utf-8

# (testing:exercise)=
# # Testing the learning environment
# 
# Before we get started, we test the learning environment and the most important packages needed to run the notebooks. This is not so much a continuous coherent exercise as individual examples based on the different packages.This exercise is neither an introduction to Python nor extensive tutorials for the individual packages. I advise you, if you have little or no experience with the packages, to work through the relevant tutorial on the websites. All packages offer very good and extensive tutorials. Most of the functions presented here have been taken from these websites.

# <div class="alert alert-block alert-success">
# <b>Learning objectives:</b><br>
# <ul>
#  <li>Getting to know the learning environment</li>
#  <li>Testing the required packages</li>
#  <li>Very brief overview of the function of the packages</li>
# </ul>  
# </div>

# <div class="alert alert-block alert-info">
# <b>How to proceed:</b><br>
# <ul>
#  <li>Testing pandas</li>
#  <li>Testing xarray</li>
#  <li>Testing MetPy</li>
# </ul>  
# </div>

# # Using pandas
# 
# Start using pandas. To load the pandas package and start working with it, import the package. The community agreed alias for pandas is pd. 
# 

# In[130]:


# Load the pandas package
import pandas as pd


# Data in Pandas is represented as a table, the so-called DataFrame. A DataFrame is a 2-dimensional data structure that can store data of different types (including characters, integers, floating point values, categorical data and more) in columns. It is similar to a spreadsheet, a SQL table or the data.frame in R. Each column in a DataFrame is a Series.

# <table><tr>
#     <td><img align="center" valign='top' src="images/01_table_dataframe.svg" height=150 >
#     <td><img align="center" valign='top' src="images/01_table_series.svg" height=150 >
# </tr></table>

# We start by reading data from a csv file into a DataFrame. pandas provides the read_csv() function to read data stored as a csv file into a pandas DataFrame. pandas supports many different file formats or data sources out of the box (csv, excel, sql, json, parquet, …), each of them with the prefix read_*.
# 

# <img align="center" valign='top' src="images/02_io_readwrite.svg" width=700 >

# Let's start and load a automatic weather station file into a pandas dataframe

# In[43]:


# Read the data into a DataFrame
df = pd.read_csv("../data/aws_valley_data_10min.csv", header=1, index_col='TIMESTAMP')


# and have a look at the dataframe

# In[35]:


# A simple way to plot the DataFrame
df.head()


# We can select a Series from the DataFrame with

# In[31]:


# Retrieve the air temperature series from the DataFrame
df['AirTC_1']


# do some calculations

# In[32]:


# Get the maximum of the air temperature series
df['AirTC_1'].max()


# As illustrated by the max() method, you can do things with a DataFrame or Series. pandas provides a lot of functionalities, each of them a method you can apply to a DataFrame or Series. As methods are functions, do not forget to use parentheses ().

# You can also get some basic statistics of the data with

# In[33]:


df.describe()


# The describe() method provides a quick overview of the numerical data in a DataFrame. Textual data is not taken into account by the describe() method.

# You can simply select specific columns from a DataFrame with

# In[47]:


# That's how you select the AirTC_1 and RH_1 columns from the df DataFrame
df_subset = df[["AirTC_1","RH_1"]]

# Plot the header (first 5 rows)
df_subset.head()


# The shape of the DataFrame can be accessed with

# In[42]:


# Access the shape attribute. Please note, do not use parentheses for attributes. 
df_subset.shape


# Often you need to filter specific rows from the DataFrame, e.g.

# <img align="center" valign='top' src="images/03_subset_rows.svg" width=700 >

# With the following command you can simply select all rows with temperatures above 5ºC

# In[54]:


# Select all rows with temerature greather than 5 degrees celsius
T_subset = df_subset[df_subset["AirTC_1"] > 5.0]

# Plot the header rows
T_subset.head()


# It is possible to combine multiple conditional statements, each condition must be surrounded by parentheses (). Moreover, you can not use or/and but need to use the or operator | and the and operator &. Here is an example

# In[60]:


# Select all rows with temerature greather than 5 degrees celsius and a relative humidity above 70%
T_RH_subset = df_subset[(df_subset["AirTC_1"] > 5.0) & (df_subset["RH_1"] > 70.0)]

# Plot the header rows
T_RH_subset.head()


# Often you want to create plots from the data.

# <img align="center" valign='top' src="images/04_plot_overview.svg" width=700 >

# To make use of the plotting function you need to load the matplotlib package

# In[62]:


# Import matplotlib
import matplotlib.pyplot as plt


# You can quickly check the data visually

# In[106]:


# Plot the temperature time series
df["AirTC_1"].plot()

# Rotate the x-labels for better readability
plt.xticks(rotation=30);


# Or create horizontally stacked plots, add two time series in one plot etc. 

# In[116]:


# Create two subplots (1 row, 2 columns)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
fig.suptitle('Horzontally stacked subplots')

# Plot the temperature time series
df["AirTC_1"].plot(ax=ax1);
# Rotate the x-labels for better readability
ax1.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right');

# Plot two temperature time series in one plot
df[["AirTC_2","AirTC_1"]].plot(ax=ax2);
# Rotate the x-labels for better readability
ax2.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right');


# Here is an example of a box plot

# In[100]:


df[["AirTC_1","RH_1","H_Flux"]].plot.box(figsize=(10,5))


# And a simple way to plot all variables in a DataFrame

# In[129]:


# Create subplots
df[["AirTC_1","RH_1","H_Flux"]].plot(figsize=(15, 5), subplots=True);

# Rotate the x-labels for better readability
plt.xticks(rotation=30);


# <div class="alert alert-block alert-info">
# <b>Note</b> This is just a starting point. Similar to spreadsheet software, pandas represents data as a table with columns and rows. Apart from the representation, also the data manipulations and calculations you would do in spreadsheet software are supported by pandas. 
# </div>

# <div class="alert alert-block alert-info">
# <b>Reminder</b> 
# <ul>
#     <li>Import the package, aka <b>import pandas as pd</b>
#     <li>A table of data is stored as a pandas DataFrame
#     <li>Each column in a DataFrame is a Series
#     <li>You can do things by applying a method to a DataFrame or Series
# </ul> 
# </div>

# <div class="alert alert-block alert-warning">
# <b>Homework:</b> Check out the pandas <a href="https://pandas.pydata.org/docs/getting_started/intro_tutorials/index.html">tutorial</a> and get familiar with the syntax.
# </div>

# # Using xarray
# 
# Multi-dimensional (a.k.a. N-dimensional, ND) arrays (sometimes called “tensors”) are an essential part of computational science. They are encountered in a wide range of fields, including physics, astronomy, geoscience, bioinformatics, engineering, finance, and deep learning. In Python, NumPy provides the fundamental data structure and API for working with raw ND arrays. However, real-world datasets are usually more than just raw numbers; they have labels which encode information about how the array values map to locations in space, time, etc.
# 
# Xarray provides a powerful and concise interface for multi-dimensional arrays (see [webpage](https://docs.xarray.dev/en/stable/index.html)). Here are some quick example of what you can do with xarray
# 

# To begin, import numpy, pandas and xarray using their customary abbreviations:

# In[34]:


# Import packages
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt


# Xarray has two core data structures, which build upon and extend the core strengths of NumPy and pandas. Both data structures are fundamentally N-dimensional:
# 
# <ul>
# <li>DataArray is our implementation of a labeled, N-dimensional array. It is an N-D generalization of a pandas.Series. 
# <li>Dataset is a multi-dimensional, in-memory array database. It is a dict-like container of DataArray objects aligned along any number of shared dimensions, and serves a similar purpose in xarray to the pandas.DataFrame.
# </ul>
# 
# In climate science we often use the NetCDF file format. You can directly read and write xarray objects to disk using to_netcdf(), open_dataset() and open_dataarray(). Suppose you have a netCDF of monthly mean data and we want to calculate the seasonal average. To do this properly, we need to calculate the weighted average considering that each month has a different number of days.
# 
# First, open the dataset

# In[21]:


# Load a netcdf dataset with xarray
ds = xr.open_dataset("data/ear5_monthly_europe.nc")


# and let's have a look to the dataset structure

# In[22]:


ds


# The dataset contains one data variable skt which has three coordinates: time, lat, and lon. We can access the coordinates very easily with

# In[23]:


# Access the time coordinates
ds.time


# We can quickly visualise the variable for a single month with

# In[25]:


# Here we plot the temperature for december 2022
ds["u10"].sel(time='2022-12-01').plot(figsize=(10,5))


# Suppose we want to calculate the seasonal average. To do this properly, we need to calculate the weighted average considering that each month has a different number of days.
# 
# We first have to come up with the weights - calculate the month length for each monthly data record 

# In[26]:


# Get the length of each monthly data record
month_length = ds.time.dt.days_in_month

# Plot the result
month_length


# Then we calculate the weights using groupby('time.season')

# In[30]:


# Calculate the weights by grouping by 'time.season'.
weights = (
    month_length.groupby("time.season") / month_length.groupby("time.season").sum()
)

# Test that the sum of the weights for each season is 1.0
np.testing.assert_allclose(weights.groupby("time.season").sum().values, np.ones(4))


# Finally, we can weight the months and sum the result

# In[31]:


# Calculate the weighted average
ds_weighted = (ds * weights).groupby("time.season").sum(dim="time")


# In[43]:


# Quick plot to show the results
notnull = pd.notnull(ds_weighted["u10"][0])

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))

ds_weighted["u10"].sel(season='DJF').where(notnull).plot.pcolormesh(
    ax=axes[0, 0],
    vmin=-10,
    vmax=10,
    cmap="Spectral_r",
    add_colorbar=True,
    extend="both",
)
ds_weighted["u10"].sel(season='MAM').where(notnull).plot.pcolormesh(
    ax=axes[0, 1],
    vmin=-10,
    vmax=10,
    cmap="Spectral_r",
    add_colorbar=True,
    extend="both",
)
ds_weighted["u10"].sel(season='JJA').where(notnull).plot.pcolormesh(
    ax=axes[1, 0],
    vmin=-10,
    vmax=10,
    cmap="Spectral_r",
    add_colorbar=True,
    extend="both",
)
ds_weighted["u10"].sel(season='SON').where(notnull).plot.pcolormesh(
    ax=axes[1, 1],
    vmin=-10,
    vmax=10,
    cmap="Spectral_r",
    add_colorbar=True,
    extend="both",
)

for ax in axes.flat:
    ax.axes.get_xaxis().set_ticklabels([])
    ax.axes.get_yaxis().set_ticklabels([])
    ax.axes.axis("tight")
    ax.set_xlabel("")

#axes[0, 0].set_title("Weighted by DPM")
#axes[0, 1].set_title("Equal Weighting")
#axes[0, 2].set_title("Difference")

plt.tight_layout()

fig.suptitle("Seasonal Surface Air Temperature", fontsize=16, y=1.02)


# You can write the results to disc

# In[45]:


# Use *.to_netcdf to the write a dataset to a netcdf file
ds_weighted.to_netcdf('weigthed_temperatures.nc')


# <div class="alert alert-block alert-info">
# <b>Reminder</b> 
# <ul>
#     <li>Import the package, aka <b>import xarray as xr</b>
#     <li>Data is stored as DataArray and Dataset
#     <li>Dataset is a multi-dimensional container of DataArray objects aligned along any number of shared dimensions, e.g. coordinates
#     <li>You can do things by applying a method to a DataArray or Dataset
#     </ul> 
# </div>

# <div class="alert alert-block alert-warning">
# <b>Homework:</b> Check out the xarray <a href="https://docs.xarray.dev/en/stable/index.html">tutorial</a> and get familiar with the syntax.
# </div>

# In[ ]:




