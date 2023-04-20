#!/usr/bin/env python
# coding: utf-8

# (nb_testing_metpy)=
# # Getting started with MetPy
# 
# Before we get started, we test the learning environment and the most important packages needed to run the notebooks. This is not so much a continuous coherent exercise as individual examples based on the different packages.This exercise is neither an introduction to Python nor extensive tutorials for the individual packages. I advise you, if you have little or no experience with the packages, to work through the relevant tutorial on the websites. All packages offer very good and extensive tutorials. Most of the functions presented here have been taken from these websites.

# <div class="alert alert-block alert-success">
# <b>Learning objectives:</b><br>
# <ul>
#  <li>Getting to know the learning environment</li>
#  <li>Testing the MetPy package</li>
#  <li>Very brief overview of the function of the packages</li>
# </ul>  
# </div>

# <div class="alert alert-block alert-info">
# <b>How to proceed:</b><br>
# <ul>
#  <li>Testing MetPy</li>
# </ul>  
# </div>

# MetPy is a collection of tools in Python for reading, visualizing, and performing calculations with weather data. One of the most significant differences in syntax for MetPy, compared to other Python libraries, is the frequent requirement of units to be attached to arrays before being passed to MetPy functions. There are very few exceptions to this, and you’ll usually be safer to always use units whenever applicable to make sure that your analyses are done correctly. Once you get used to the units syntax, it becomes very handy, as you never have to worry about unit conversion for any calculation. MetPy does it for you!
# 
# Let's load the MetPy and numpy library
# 

# In[1]:


# Load the pandas package
import numpy as np
from metpy.units import units


# Please note, we have loaded only the units module here. We can assign a unit to an array by

# In[2]:


# Attach the unit meters to the distance array
distance = np.arange(1, 5) * units.meters


# Similarly, we can attach create a time array

# In[3]:


# This is another way to attach units to an array 
time = units.Quantity(np.arange(2, 10, 2), 'sec')


# Now, we can simply do unit-aware calculations

# In[4]:


# Calculate the velocity
distance/time


# ## MetPy with xarray
# 
# MetPy works great with xarray. MetPy’s suite of meteorological calculations are designed to integrate with xarray DataArrays and provides DataArray and Dataset accessors (collections of methods and properties attached to the .metpy property) for coordinate/CRS and unit operations.
# 
# First, some imports ...

# In[5]:


# Import xarray
import xarray as xr

# Any import of metpy activates the accessors
import metpy
from metpy.cbook import get_test_data


# We can open a netCDF-file using xarray

# In[6]:


# Open the netCDF file as a xarray Dataset
ds = xr.open_dataset(get_test_data('irma_gfs_example.nc', as_file_obj = False))

# View a summary of the Dataset
ds


# This Dataset consists of dimensions and their associated coordinates, which in turn make up the axes along which the data variables are defined. The dataset also has a dictionary-like collection of attributes. What happens if we look at just a single data variable?

# In[7]:


temperature = ds['Temperature_isobaric']
temperature


# This is a DataArray, which stores just a single data variable with its associated coordinates and attributes. These individual DataArrays are the kinds of objects that MetPy’s calculations take as input (more on that in Calculations section below).

# ## Coordinates and Corrdinate Reference Systems

# MetPy’s first set of helpers comes with identifying coordinate types. In a given dataset, coordinates can have a variety of different names (time, isobaric1 ...). Following CF conventions, as well as using some fall-back regular expressions, MetPy can systematically identify coordinates of the following types:
# 
# <ul>
#     <li>time
#     <li>vertical
#     <li>latitude
#     <li>y
#     <li>longitude
#     <li>x
# </ul>
# 
# When identifying a single coordinate, it is best to use the property directly associated with that type
# 

# In[8]:


temperature.metpy.time


# In[9]:


x, y = temperature.metpy.coordinates('x', 'y')
x


# These coordinate type aliases can also be used in MetPy’s wrapped .sel and .loc for indexing and selecting on DataArrays. For example, to access 500 hPa heights at 1800Z,

# In[10]:


heights = ds['Geopotential_height_isobaric'].metpy.sel(
    time='2017-09-05 18:00',
    vertical=50000.
)
heights


# Beyond just the coordinates themselves, a common need for both calculations with and plots of geospatial data is knowing the coordinate reference system (CRS) on which the horizontal spatial coordinates are defined. MetPy follows the CF Conventions for its CRS definitions, which it then caches on the metpy_crs coordinate in order for it to persist through calculations and other array operations. There are two ways to do so in MetPy:
# 
# First, if your dataset is already conforming to the CF Conventions, it will have a grid mapping variable that is associated with the other data variables by the grid_mapping attribute. This is automatically parsed via the .parse_cf() method:

# In[11]:


# Parse full dataset
data_parsed = ds.metpy.parse_cf()

# Parse subset of dataset
data_subset = ds.metpy.parse_cf([
    'u-component_of_wind_isobaric',
    'v-component_of_wind_isobaric',
    'Vertical_velocity_pressure_isobaric'
])

# Parse single variable
relative_humidity = ds.metpy.parse_cf('Relative_humidity_isobaric')


# Notice the newly added metpy_crs non-dimension coordinate. Now how can we use this in practice? For individual DataArrays, we can access the cartopy and pyproj objects corresponding to this CRS:

# In[12]:


# Cartopy CRS, useful for plotting
relative_humidity.metpy.cartopy_crs


# In[13]:


# pyproj CRS, useful for projection transformations and forward/backward azimuth and great
# circle calculations
relative_humidity.metpy.pyproj_crs


# ## Units
# 
# Since unit-aware calculations are a major part of the MetPy library, unit support is a major part of MetPy’s xarray integration!
# 
# One very important point of consideration is that xarray data variables (in both Datasets and DataArrays) can store both unit-aware and unit-naive array types. Unit-naive array types will be used by default in xarray, so we need to convert to a unit-aware type if we want to use xarray operations while preserving unit correctness. MetPy provides the .quantify() method for this (named since we are turning the data stored inside the xarray object into a Pint Quantity object)

# In[15]:


temperature = data_parsed['Temperature_isobaric'].metpy.quantify()
temperature


# Notice how the units are now represented in the data itself, rather than as a text attribute. Now, even if we perform some kind of xarray operation (such as taking the zonal mean), the units are preserved.

# In[ ]:


# Take the mean over time at the 1000 hPa level
temperature.sel(isobaric3=1000).mean('time1')


# # Plotting 
# 
# 

# In[ ]:


# Load data
ds = xr.open_dataset(get_test_data('narr_example.nc', as_file_obj = False))

# Parse full dataset
ds = ds.metpy.parse_cf()

# Grab lat/lon values from file as unit arrays
lats = ds.lat.metpy.unit_array
lons = ds.lon.metpy.unit_array

# Get the valid time
vtime = ds.Temperature.metpy.time[0]

# Get the 700-hPa heights without manually identifying the vertical coordinate
hght_700 = ds.Geopotential_height.metpy.sel(vertical=700 * units.hPa,
                                                 time=vtime)


# In[ ]:


# Cartopy CRS, useful for plotting
hght_700.metpy.cartopy_crs


# In[ ]:


# Import cartopy for plotting
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt

# Open figure
fig = plt.figure(figsize=(12, 12))

# Add a plot with coordinate axes (crs)
ax = fig.add_subplot(1, 1, 1, projection=hght_700.metpy.cartopy_crs)

# Get coordinates
x = ds.x
y = ds.y

# Plot data
ax.imshow(hght_700, extent=(x.min(), x.max(), y.min(), y.max()),
          cmap='RdBu', origin='lower' if y[0] < y[-1] else 'upper')

# Add coastlines
ax.coastlines(color='tab:blue', resolution='10m')

# Show the plot
plt.show()


# ## Calculations

# In[ ]:


import numpy as np
from metpy.units import units
import metpy.calc as mpcalc


# In[ ]:


temperature = [20] * units.degC
rel_humidity  = [50] * units.percent
print(mpcalc.dewpoint_from_relative_humidity(temperature, rel_humidity))


# In[ ]:


speed = np.array([5, 10, 15, 20]) * units.knots
direction = np.array([0, 90, 180, 270]) * units.degrees
u, v = mpcalc.wind_components(speed, direction)
print(u, v)


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
