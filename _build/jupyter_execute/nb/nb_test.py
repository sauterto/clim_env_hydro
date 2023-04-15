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

# # Pandas
# 
# Start using pandas. To load the pandas package and start working with it, import the package. The community agreed alias for pandas is pd. 
# 

# In[2]:


# Load the pandas package
import pandas as pd


# Data in Pandas is represented as a table, the so-called DataFrame. A DataFrame is a 2-dimensional data structure that can store data of different types (including characters, integers, floating point values, categorical data and more) in columns. It is similar to a spreadsheet, a SQL table or the data.frame in R. Each column in a DataFrame is a Series.

# <img align="left" valign='top' src="images/01_table_dataframe.svg" height=150 >
# <img align="left" valign='top' src="images/01_table_series.svg" height=150 >

# We start by reading data from a csv file into a DataFrame. pandas provides the read_csv() function to read data stored as a csv file into a pandas DataFrame. pandas supports many different file formats or data sources out of the box (csv, excel, sql, json, parquet, …), each of them with the prefix read_*.
# 

# <img align="left" valign='top' src="images/02_io_readwrite.svg" height=150 >

# Let's start and load a automatic weather station file into a pandas dataframe

# In[29]:


# Read the data into a DataFrame
df = pd.read_csv("../data/aws_valley_data_10min.csv", header=1, index_col='TIMESTAMP')


# and have a look at the dataframe

# In[30]:


# A simple way to plot the DataFrame
df


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

# <div class="alert alert-block alert-info">
# <b>Note</b> This is just a starting point. Similar to spreadsheet software, pandas represents data as a table with columns and rows. Apart from the representation, also the data manipulations and calculations you would do in spreadsheet software are supported by pandas. Continue reading the next tutorials to get started!  
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

# ## Download station list
# 
# Read the station list into pandas DataFrame (from file igra2-station-list.txt in the IGRAv2 repository). In case you are not familiar with pandas, please check out the pandas webpage \[[link](https://pandas.pydata.org)\] 

# In[135]:


# Load the IGRAv2 radiosonde tools
import igra

# Load pandas
import pandas

# Get the station list and store it in the tmp folder
stations = igra.download.stationlist('./tmp')


# In[136]:


# Have a look at the data
stations


# <div class="alert alert-block alert-warning">
# <b>Tip:</b> Check out the IGRA webpage to see all available stations
# </div>

# ## Download station
# 
# Download a radiosonde station with the *id* from the station list into tmp directory.

# In[137]:


id = "GMM00010868"
igra.download.station(id, "./tmp")


# <div class="alert alert-block alert-warning">
# <b>Exercise:</b> What is the ID of the INNSBRUCK-FLUGHAFEN radiosonde station?
# </div>

# ## Read station data
# 
# The downloaded station file can be read to standard pressure levels (default). In case you prefer to download all significant levels (different amount of levels per sounding) use return_table=True. 

# In[201]:


data, station = igra.read.igra(id, "./tmp/<id>-data.txt.zip".replace('<id>',id), return_table=True)


# Have a look at the data

# In[202]:


data


# <div class="alert alert-block alert-warning">
# <b>Exercise:</b> Find out what the individual variables are and in which unit they are given.
# </div>

# <div class="alert alert-block alert-warning">
# <b>Advanced exercise:</b> Plot the vertical temperature profile and determine the stratification of the atmosphere at different altitudes.
# </div>

# ## Thermodynamic Calculations
# 
# MetPy is a collection of tools in Python for reading, visualizing, and performing calculations with weather data  [[Link](https://unidata.github.io/MetPy/latest/index.html)]. Here, we use the MetPy calc module to calculate some thermodynamic parameters of the sounding.
# 
# **Lifting Condensation Level (LCL)** - The level at which an air parcel’s relative humidity becomes 100% when lifted along a dry adiabatic path.
# 
# **Parcel Path** - Path followed by a hypothetical parcel of air, beginning at the surface temperature/pressure and rising dry adiabatically until reaching the LCL, then rising moist adiabatially.

# In[203]:


# Load the metpy package. MetPy is a collection of tools 
# in Python for reading, visualizing, and performing calculations 
# with weather data. 

# Module to work with units
from metpy.units import units

# Collection of calculation function
import metpy.calc as mpcalc

# Import the function to plot a skew-T diagram
from metpy.plots import SkewT


# <div class="alert alert-block alert-warning">
# <b>Exercise:</b> Check out the MetPy website and familiarise yourself with the collection of meteorological functions. Which functions can be used to calculate the LCL and the parcel path?
# </div>

#    We pre-process the balloon data to meet the requirements of the MetPy functions. 

# In[228]:


# For which day should the calculations be carried out?
timestamp = '2022-08-15T12:00'

# Select the corresponding dataset
data_subset = data.sel(date=timestamp)

# Here, the variables are prepared and units are assigned to the values
# Temperature data in degree celcius
T = (data_subset.temp.values-273.16) * units.degC

# Dewpoint temperature in degree celcius
Td = T - data_subset.dpd.values * units.delta_degC

# Wind speed in meter per second
wind_speed = data_subset.winds.values * units('m/s')

# Wind direction in degrees
wind_dir = data_subset.windd.values * units.degrees

# Pressure in Hektapascal
p = (data_subset.pres.values/100) * units.hPa

# Since MetPy assumes the arrays from high to lower pressure, 
# but the IGRA data is given from low to high pressure, the 
# arrays must be reversed.
p = p[~np.isnan(T)][::-1]
T = T[~np.isnan(T)][::-1]
Td = Td[~np.isnan(Td)][::-1]
wind_speed = wind_speed[~np.isnan(wind_speed)][::-1]
wind_dir = wind_dir[~np.isnan(wind_dir)][::-1]


# <div class="alert alert-block alert-warning">
# <b>Exercise:</b> Why was the <em>dpd</em> variable subtracted from the temperature to get the dew point temperature?
# </div>

# <div class="alert alert-block alert-warning">
# <b>Exercise:</b> What is achieved with the statement ~np.isnan()[::-1]?
# </div>

# With the following command, the wind components can be calculated from the wind speed and wind direction.
# 

# In[223]:


u, v = mpcalc.wind_components(wind_speed, wind_dir)


# <div class="alert alert-block alert-warning">
# <b>Exercise:</b> Refer to the MetPy reference to see how the wind direction and speed can be calculated from the wind components.
# </div>

# Finally, the LCL and parcel profile can be calculated with the pre-processed data.

# In[224]:


# Calculate the LCL
lcl_pressure, lcl_temperature = mpcalc.lcl(p[0], T[0], Td[0])

# Calculate the parcel profile
parcel_prof = mpcalc.parcel_profile(p, T[0], Td[0]).to('degC')

print('LCL pressure level: {:.2f}'.format(lcl_pressure))
print('LCL temperatur: {:.2f}'.format(lcl_temperature))


# With the calculated and processed data we can finally create the Skew-T diagram.

# In[229]:


# Create a new figure. The dimensions here give a good aspect ratio
fig = plt.figure(figsize=(12, 12))
skew = SkewT(fig, rotation=30)

# Plot the data using normal plotting functions, in this case using
# log scaling in Y, as dictated by the typical meteorological plot
skew.plot(p, T, 'r')
skew.plot(p, Td, 'g')
skew.plot_barbs(p, u, v)
skew.ax.set_ylim(1000, 100)
skew.ax.set_xlim(-40, 40)

# Plot LCL temperature as black dot
skew.plot(lcl_pressure, lcl_temperature, 'ko', markerfacecolor='black')

# Plot the parcel profile as a black line
skew.plot(p, parcel_prof, 'k', linewidth=2)

# Shade areas of CAPE and CIN
skew.shade_cin(p, T, parcel_prof, Td)
skew.shade_cape(p, T, parcel_prof)

# Plot a zero degree isotherm
skew.ax.axvline(0, color='c', linestyle='--', linewidth=2)

# Add the relevant special lines
skew.plot_dry_adiabats(linewidth=1)
skew.plot_moist_adiabats(linewidth=1)
skew.plot_mixing_lines(linewidth=1)

# Show the plot
plt.show()


# <div class="alert alert-block alert-warning">
# <b>Exercise:</b> Take a close look at the diagram and try to understand the structure. 
# <ul>
# <li>What do the different coloured lines show? 
# <li>What is the shaded area?  
# <li>Which areas are stable or unstable stratified? 
# <li>Where does condensation take place? 
# </ul>
# </div>
