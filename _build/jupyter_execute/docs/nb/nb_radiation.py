#!/usr/bin/env python
# coding: utf-8

# (nb_radiation)=
# # Radiation and ground heat flux
# 
# While large-scale weather shapes environmental conditions, energy exchange in mountain valleys is controlled by micrometeorological conditions. Given the complex topography with its different surface properties, it is not trivial to close the scale gap between the large-scale conditions and the local properties. The micrometeorological state of the surface layer is directly influenced by the Earth's surface and responds rapidly to changes in the surface energy budget. Radiative and turbulent heat fluxes cool and heat the near-surface air layer and determine the temperature distribution over the topography. Local temperature surpluses and deficits generate upwelling forces that drive thermal wind systems, including valley circulations, slope and glacier winds
# 

# <div class="alert alert-block alert-success">
# <b>Learning objectives:</b><br>
# <ul>
#  <li>Parametrization of radiation fluxes</li>
#  <li>Ground heat flux</li>
# </ul>  
# </div>

# <div class="alert alert-block alert-info">
# <b>Prerequisites</b><br>
# <ul>
# <li>Basic knowledge of Python, Jupyter Notebooks, and data analysis</li>
# <li>Familiarity with MetPy, Pandas, Xarray, and Plotly</li>
#     <li>The additional package <b>xrspatial</b> must be installed
# <li>A netcdf file with the shadings and sky-view factors (can be downloaded <a href="https://box.hu-berlin.de/f/fad79e62bbfe4403a604/?dl=1" download>here</a>)
# </ul>  
# </div>

# In[ ]:


import numpy as np
import pandas as pd
import math
import xarray as xr

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
#-----------------------------
hu = HU_learning_material()


# The functions can then be accessed with **hu***.function()*, where the placeholder *function* stands for the various functions available. With the following command, all functions/methods within the class can be displayed.

# In[ ]:


class_methods = [method for method in dir(hu)
                 if not method.startswith('__')
                 and callable(getattr(hu, method))
                 ]
print(class_methods) 


# You can find out more about the function/method and its arguments with

# In[ ]:


# Example: Information about the emissivity() method
help(hu.emissivity)


# # Solar parameters and transmissivity

# We start with solar paramters and transmissivity (see slides). To calculate the local elevation angle of the sun, we need the coordinates of the location and the time.

# In[ ]:


#----------------------
# Site
#----------------------
lat = 45.0  # Latitude of location
lon = 10.0  # Longitud of location
UTC = 19.5  # UTC at location


# First, we create a time vector for the period for which we want to calculate the local sun angle.

# In[ ]:


# Create a date vector for which the solar parameters and transmisivity are calculated
timestamp = pd.date_range(start='01-01-2002', end='31-12-2002', freq='D')


# Then we can calculate the angle with the function **local_elevation_angle()**. The function returns the sine of the angle (see slides). To obtain the final angle (radian), the arc sine of this value must be taken. Finally, we convert the radian into degrees with the function **rad2deg()**.

# In[ ]:


# Local elevation angle using th HU-Tools
angle = np.rad2deg(np.arcsin(hu.local_elevation_angle(lat,lon,UTC,timestamp.dayofyear,173,timestamp.size)))


# For the calculation of the transmissivity, the proportions of the cloud cover in the different cloud levels are required. 

# In[ ]:


#----------------------
# Parameters
#----------------------
ch = 0.0    # High cloud fraction
cm = 0.0    # Middle cloud fraction
cl = 0.0    # Low cloud fraction


# This allows the transmissivity at a given place and time to be approximated using the function **tau()**.

# In[ ]:


# Calculate transmissivity using th HU-Tools
data  = hu.tau(ch, cm, cl, lat, lon, UTC, timestamp.dayofyear, 173, timestamp.size)


# To facilitate working with the results, we write the results into a Pandas DataFrame.

# In[ ]:


# Create a pandas DataFrame from the calculated arrays and the date vector
df = pd.DataFrame(data, index=timestamp, columns=['tau'])

# Add the local elevation angle to the DataFrame
df['angle'] = angle


# Plotly is a good way to display the data.

# In[ ]:


#----------------------
# Plot the results
#----------------------
# Creating the plot with two rows and one column. The plots share the same x-axis so that only labels 
# for the lower plots are shown
fig = make_subplots(rows=2, cols=1, shared_xaxes=True)

# Adding the transmissivity data as a  line in the top panel
fig.add_trace(go.Scatter(x=df.index, y=df.tau, line=dict(color='royalblue', dash='solid'), name='Transmissitivty [-]'),
                        row=1, col=1)

# Adding the local elevation angle as line in the bottom panel
fig.add_trace(go.Scatter(x=df.index, y=df.angle, line=dict(color='green', dash='solid'), name='Local elevation angle [º]'),
                         row=2, col=1)

# Adjusting the layout
fig.update_layout(title='Transmissivity and local elevation angle', plot_bgcolor='white', width=800, height=600,
                  yaxis=dict(title='Transmissitivty [-]', showgrid=True, gridcolor='lightgray', gridwidth=1),
                  yaxis2=dict(title='Local elevation angle [º]', showgrid=True, gridcolor='lightgray', gridwidth=1),
                  xaxis=dict(title='', tickformat='%d.%m.%Y', showgrid=True, gridcolor='lightgray', gridwidth=1),
                  xaxis2=dict(title='Date', tickformat='%d.%m.%Y', showgrid=True, gridcolor='lightgray', gridwidth=1))


# Adjusting the axes
fig.update_xaxes(nticks=10, row=1, col=1)
fig.update_yaxes(nticks=5, row=1, col=1)
fig.update_xaxes(nticks=10, row=2, col=1)
fig.update_yaxes(nticks=5, row=2, col=1)

# Showing the plot
fig.show()


# <div class="alert alert-block alert-warning">
# <b>Broaden Knowledge & Comprehension</b></br>
# In what period of time is the sun still above the horizon at 45º north and 10º east after 7:30pm?
# How sensitive is the transmissivity to clouds in the low, middle and high atmosphere? Look at the equation for transmissivity and find out why there are differences. What is a weakness of the parameterisation?
# </div>

# # Shortwave radiation and albedo

# Most weather stations measure the short-wave irradiation directly.  If the albedo of the surfaces is known, the short-wave radiation balance can be calculated very easily.

# In[ ]:


albedo = 0.7  # [-] 
SWin = 342    # W/mˆ2

print('Incoming shortwave radiation: {:.2f}'.format(SWin))
print('Outgoing shortwave radiation: {:.2f}'.format(hu.SWout(SWin,albedo)))
print('Net shortwave radiation: {:.2f}'.format(SWin-hu.SWout(SWin,albedo)))


# <div class="alert alert-block alert-warning">
# <b>Broaden Knowledge & Comprehension</b></br>
# Review the 'Global Energy Fluxes' figure on slide 4 of the presentation and identify the outgoing shortwave radiation. Use this information to determine the global albedo.
# 
# Here the value 342 $W/m^{2}$ was used, which corresponds to the average global incoming solar radiation. Why is this value lower than the solar constant, which has a value of ~1362 $W/m^{2}$?
# </div>
# 

# In many cases it is necessary to calculate the direct and diffuse radiation in complex terrain. In order to calculate the radiation fluxes, the shading effects must be taken into account. In this section we briefly discuss how this can be done.
# 
# First, we load the digital elevation model. The georeferenced Geotiff has already been saved here in netcdf format.

# In[ ]:


#------------------------------
# Read digital elevation model
#------------------------------
dem = xr.open_dataset('./data/paine_elevation.nc')
dem


# The file contains the variable band_data which contains the height information. The dimension of the variable is band_data(x, y) where x corresponds to the longitude values and y to the latitude values. We can now read the 2D data from the variable *band_data* from the file and do a quick plot.

# In[ ]:


# Read the elevation data from netcdf file
elvgrid = dem['band_data']


# For the calculation of the shading effects, the slope and aspect of the cells must also be specified. For this you can use the library **xrspatial** which provides the two functions **slope()** and **aspect()**. The terrain model is passed to the two functions and the corresponding grid is returned.

# In[ ]:


# Load the xrspatial library
from xrspatial import slope, aspect

# Read slope data
slopegrid = slope(elvgrid)
# Read aspect data
aspectgrid = aspect(elvgrid)


# Again, use plotly to plot the data

# In[ ]:


#---------------------------
# Plot data
#---------------------------
# Generate sample data
x = lons
y = lats
X, Y = np.meshgrid(x, y)

# Create the subplots
fig = sp.make_subplots(rows=1, cols=3)

# Add the first subplot
fig.add_trace(go.Contour(x=x, y=y, z=slopegrid, colorscale='Viridis', showscale=True,
                         colorbar=dict(x=0.12, y=-0.35, len=0.2, yanchor='bottom', orientation='h',
                                      title=r'Slope in degrees', titleside='top')), row=1, col=1)

# Add the second subplot
fig.add_trace(go.Contour(x=x, y=y, z=aspectgrid, colorscale='YlOrRd', showscale=True,
                         colorbar=dict(x=0.5, y=-0.35, len=0.2, yanchor='bottom', orientation='h', 
                                       title=r'Aspect in degrees north', titleside='top')), row=1, col=2)

# Add the second subplot
fig.add_trace(go.Contour(x=x, y=y, z=elvgrid, colorscale='YlOrRd', showscale=True,
                         colorbar=dict(x=0.85, y=-0.35, len=0.2, yanchor='bottom', orientation='h', 
                                       title=r'Digital elevation model', titleside='top')), row=1, col=3)

# Add title and axis labels
fig.update_layout(
    xaxis=dict(title='Longitude',title_standoff=5),
    xaxis2=dict(title='Longitude',title_standoff=5),
    yaxis_title='Latitude',
    width=1200,  # Set the width of the figure
    height=600  # Set the height of the figure
)

# Show the plot
fig.show()


# For simplicity, the shading and the sky-view factor for the terrain model have already been calculated and written into the file LUT_Rad.nc. However, the two functions **LUTshad()** and **LUTsvf()** are available with which both can be calculated (caution: the calculations take a very long time depending on the DEM). We load the sky-view factor and the shading for a complete in the variables *svf* and *shad1yr*.

# In[ ]:


# Load the dataset
ds_LUT = xr.open_dataset('LUT_Rad.nc')

# Read the shading field for a complete year
shad1yr = ds_LUT.SHADING.values

# Read the sky-view factor
svf = ds_LUT.SVF.values


# Direct and diffuse radiation both depend on several variables: Temperature, pressure, humidity and cloudiness. We assume that the same conditions prevail throughout the domain. To do this, we create new 2D arrays with the corresponding values. 

# In[ ]:


# Create additional arrays which are needed by the radiation module
# Temperature grid
tempgrid = np.ones_like(elvgrid) * 285.0
# Pressure grid
pgrid = np.ones_like(elvgrid) * 1000.0
# Relative humditiy grid
rhgrid = np.ones_like(elvgrid) * 70
# Cloud grid
ngrid = np.ones_like(elvgrid) * 0.0
# Mask grid (in case only part of the DEM is used
maskgrid = xr.zeros_like(elvgrid)


# Now we can use this information to calculate the radiation fluxes.

# In[ ]:


# Site information
lat = -50.9 # Latitude of the domain center
doy = 15    # Doy of the year
hour = 14   # Hour of the day
tcart = 0   # Time shifts [do not change]
dtstep = 3*3600 # Time step between the shading values [do not change]

# Get lon/lat values from the elevation model
lons = elvgrid.x
lats = elvgrid.y

#---------------------------
# Calculate solar Parameters
#---------------------------
solPars, timeCorr = hu.solpars(lats[0])

#---------------------------
# Calc radiation
#---------------------------
A = hu.calcRad(solPars, timeCorr, doy, hour, lat, tempgrid[::-1, :], pgrid[::-1, :], rhgrid[::-1, :], 
        ngrid[::-1, :], np.flipud(elvgrid), np.flipud(maskgrid), np.flipud(slopegrid), np.flipud(aspectgrid), shad1yr, svf, dtstep, tcart)


# Plot the fields

# In[ ]:


#---------------------------
# Plot data
#---------------------------
# Generate sample data
x = lons
y = lats
X, Y = np.meshgrid(x, y)

# Create the subplots
fig = sp.make_subplots(rows=1, cols=2)

# Add the first subplot
fig.add_trace(go.Contour(x=x, y=y, z=elvgrid, colorscale='Viridis', showscale=True,
                         colorbar=dict(x=0.2, y=-0.22, len=0.4, yanchor='bottom', orientation='h',
                                      title=r'$\text{Incoming shortwave radiation}~\text{W} m^{-2}$', titleside='top')), row=1, col=1)

# Add the second subplot
fig.add_trace(go.Contour(x=x, y=y, z=A, colorscale='YlOrRd', showscale=True,
                         colorbar=dict(x=0.8, y=-0.22, len=0.4, yanchor='bottom', orientation='h', 
                                       title=r'$\text{Incoming shortwave radiation}~\text{W} m^{-2}$', titleside='top')), row=1, col=2)



# Add title and axis labels
fig.update_layout(
    xaxis=dict(title='Longitude',title_standoff=5),
    xaxis2=dict(title='Longitude',title_standoff=5),
    yaxis_title='Latitude',
    width=1200,  # Set the width of the figure
    height=600  # Set the height of the figure
)

# Show the plot
fig.show()


# <div class="alert alert-block alert-warning">
# <b>Broaden Knowledge & Comprehension</b></br>
# Which areas receive the highest levels of radiation? How does the field change during the day? What would the field look like with the same topography in the northern hemisphere?
# </div>
# 

# # Longwave radiation

# Long-wave radiation fluxes are important terms in the net radiation balance. The long-wave radiation based on the Stephan-Boltzmann equation can be calculated with the function **LW()** of the **HU_learning_material** class. It is also possible to calculate the wavelength of the emission maximum with Kirchoff's law using the **lambda_max()** function.
# 
# Let's assume the emissivity of the body is 1.0 ...

# In[ ]:


# Define a emissivity
epsilon = 1.0

# Radiated energy at zero degrees celcius
print('Radiated energy at zero degrees celcius: {:.4}'.format(hu.LW(epsilon, 273)))

# Wave length of emission maximum
print('Wave length of emission maximum: {:.4}'.format(hu.lambda_max(273)))


# <div class="alert alert-block alert-success">
# <b>Evaluate and Analyze</b><br>
# <ul>
# <li>Calculate the long-wave radiation for different temperatures. What is the relationship between temperature and radiated energy? [Hint: Take a look at the Stephan-Boltzmann Law]
# <li>Plot the wavelength of the emission maximum against the radiated energy in the range from 250 K to 310 K. Which gases absorb in this spectral range? Comment on the influence of the most important gases on the greenhouse effect?
# <li>Compare the values with those of the short-wave radiation. What are the most important radiation terms on a summer day in the mountains?
# </ul>
# </div>

# <div class="alert alert-block alert-warning">
# <b>Broaden Knowledge & Comprehension</b></br>
# What must be taken into account when measuring long-wave radiation in order to obtain reliable values?
# </div>

# # Ground heat flux and heat equation

# The radiation terms are the largest in terms of magnitude. However, the soil heat flux, although much smaller, also plays an important role in the energy balance of a surface. The soil heat flux is coupled to the soil temperature profile. The function **heat_equation()** approximates the temperature profile. This requires to define some boundary conditions, such as the temperature at the surface or at bottom. In addition, information on the model depth, integration time, time step and soil thermal diffusivity is required.

# In[ ]:


#-----------------------
# Boundary conditions
#-----------------------
Ts = 20.   # Temperature at the surface [K]
Tb = 5.    # Temperature at the bottom [K]
D  = 2.    # Depth [m]
Nz = 100   # Number of grid points in the vertical (soil)
dt = 60.   # time step [s]
t7200  = 7200  # integration time [s]
t86400 = 86400 # integration time [s]
alpha = 1.2e-6 # soil thermal diffusivity [m^2 s^-1]


# We now run the model for two different integration periods and plot both temperature profiles

# In[ ]:


#-----------------------
# Run the heat equation
#-----------------------
T, dz = hu.heat_equation(Ts, Tb, D, Nz, t7200, dt, alpha)
T2, dz = hu.heat_equation(Ts, Tb, D, Nz, t86400, dt, alpha)

#-----------------------
# Make plots
#-----------------------
# Creating the plot with two rows and one column. The plots share the same x-axis so that only labels 
# for the lower plots are shown
fig = make_subplots(rows=1, cols=1, shared_yaxes=True)

# Adding the temperature data as a dashed line in the top panel
fig.add_trace(go.Scatter(x=T, y=-dz*np.arange(Nz), line=dict(color='royalblue', dash='solid'), name='Temperature after 7200 s'),
                        row=1, col=1)

# Adding the temperature data as a dashed line in the top panel
fig.add_trace(go.Scatter(x=T2, y=-dz*np.arange(Nz), line=dict(color='green', dash='solid'), name='Temperature after 86400'),
                        row=1, col=1)

# Adjusting the layout
fig.update_layout(title='Soil temperature', plot_bgcolor='white', width=800, height=600,
                  yaxis=dict(title='Depth [m]', showgrid=True, gridcolor='lightgray', gridwidth=1),
                  xaxis=dict(title='Temperature [K]', tickformat='%d.%m.%Y', showgrid=True, gridcolor='lightgray', gridwidth=1))


# Adjusting the axes
fig.update_xaxes(nticks=10, row=1, col=1, range=[-2,20], dtick=5)
fig.update_yaxes(nticks=5, row=1, col=1)

# Showing the plot
fig.show()


# <div class="alert alert-block alert-warning">
# <b>Broaden Knowledge & Comprehension</b></br>
# <ul>
# <li>How can the ground heat flux be determined from the temperature profile? 
# <li>Derive the ground heat flux from the figure. 
# <li>Write a function to calculate the soil heat flux.
# </ul>
# </div>

# In nature, the surface temperature is determined by the energy fluxes. We can simply modify the model to allow for varying surface temperature. We create a Pandas DataFrame with an artificial temperature time series with daily mean temperatures. 

# In[ ]:


#-----------------------------
# Create pandas dataframe with
# the surface temperature
#-----------------------------
# timestamp from/to
timestamp = pd.date_range(start='2000-01-01T00:00:00', end='2003-12-31T00:00:00', freq='D')

# Surrogate surface temperature timeseries
T = 10 - 20 * np.sin((2*math.pi*timestamp.dayofyear)/365)

# Create a pandas DataFrame from the calculated arrays
df = pd.DataFrame(T, index=timestamp, columns=['T'])


# In the next step, we set the boundary conditions and run the heat_equation_daily()

# In[ ]:


#-----------------------------
# Run the time-dependent heat
# equation
#-----------------------------
D  = 20   # Depth of the domain in meter
dz = 0.5  # Spacing between grid points in meter
T, D, dz = hu.heat_equation_daily(df['T'], D, dz, alpha=1.2e-6)


# Finally, we plot the results

# In[ ]:


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
    z=T[::-1],
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
    xaxis=dict(title='Date', tickformat='%d.%m.%Y', tickangle=45, dtick='M3'),
    yaxis=dict(title='Depth [m]'),
    width=1000, height=500,
)

# Create the figure
fig = go.Figure(data=[trace], layout=layout)

# Show the plot
fig.show()


# <div class="alert alert-block alert-warning">
# <b>Broaden Knowledge & Comprehension</b></br>
# <ul>
# <li> Change the soil thermal diffusivity and see how the profile changes (Use realistic values of different soil types) 
# <li> Describe how the temperature profile changes over time
# <li> Integrate the model over a few decades and find out how long it takes to reach a steady-state (stable temperature profile).
# </ul>
# </div>
# 

# The amplitude of the temperature fluctuations decreases with depth. Let's plot the temperatures at 5 and 10 m depth.

# In[ ]:


# Creating the plot with two rows and one column. The plots share the same x-axis so that only labels 
# for the lower plots are shown
fig = make_subplots(rows=1, cols=1, shared_xaxes=True)

# Adding the temperature data as a dashed line in the top panel
fig.add_trace(go.Scatter(x=df.index, y=T[10,:], line=dict(color='royalblue', dash='solid'), name='5 m'),
                        row=1, col=1)

# Adding the temperature data as a dashed line in the top panel
fig.add_trace(go.Scatter(x=df.index, y=T[20,:], line=dict(color='orange', dash='solid'), name='10 m'),
                        row=1, col=1)

# Adjusting the layout
fig.update_layout(title='Soil temperature', plot_bgcolor='white', width=1000, height=600,
                  yaxis=dict(title='Soil temperature [K]', showgrid=True, gridcolor='lightgray', gridwidth=1),
                  xaxis=dict(title='', tickformat='%d.%m.%Y', showgrid=True, gridcolor='lightgray', gridwidth=1))


# Adjusting the axes
fig.update_xaxes(nticks=10, row=1, col=1, tickformat='%d.%m.%Y', tickangle=45, dtick='M3')
fig.update_yaxes(nticks=5, row=1, col=1)

# Showing the plot
fig.show()


# <div class="alert alert-block alert-warning">
# <b>Broaden Knowledge & Comprehension</b></br>
# <ul>
# <li> What can you observe?
# <li> Describe the differences between the time series.
# <li> How would the time series change if the thermal diffusivity is increased? Why?
# </ul>
# </div>

# ## More realistic example
# 
# In this example, the code reads in the 30-min weather stations data from a CSV file using the '**read_csv()**' method from pandas, with the parse_dates and index_col parameters set to True and 0, respectively. This ensures that the first column of the CSV file (the timestamps) is used as the index of the DataFrame, and that the timestamps are converted to datetime objects.

# In[ ]:


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


# <div class="alert alert-block alert-warning">
# <b>Broaden Knowledge & Comprehension</b></br>
# Which other file formats can be read in with pandas?
# </div>

# ## Replace missing data
# 
# Next, we want to view the data to ensure it has been loaded correctly. We use the head() method from Pandas to display the first five rows of the DataFrame.

# The DataFrame has missing values with the dummy value -9999. We then use the replace method to replace all occurrences of -9999 with NaN (Not a Number), using the NumPy np.nan constant. Finally, we print the updated DataFrame. The inplace=True argument ensures that the original DataFrame is modified, rather than a copy being created.

# In[ ]:


# replace -9999 with NaN
df.replace(-9999, np.nan, inplace=True)

# Keep only the surface temeperature
ds = df['TS']

# check for missing values
ds.dropna(inplace=True)

# Show time series
ds


# <div class="alert alert-block alert-warning">
# <b>Hint!</b> Compare the SW_OUT and LW_OUT column with the table before the missing values have been removed.
# </div>

# To check for missing values in a Pandas DataFrame, you can use the **isna()** or **isnull()** method, which returns a Boolean DataFrame of the same shape indicating which cells contain missing values.

# You can also use the isna() or isnull() method along with the **sum()** method to count the number of missing values in each column:

# In[ ]:


# resample dataframe to monthly mean values with different aggregation for each column and return NaN for 
# columns with insufficient valid elements
threshold =  50 # What is the maximum percentage of NaNs that may be included in the averaging? Here, we set the threshold to 10%

# This is a pythonic way in solving this problem
ds_daily = ds.resample('1D').agg({'TS': lambda x: x.dropna().mean() if (((x.isna().sum())/len(x))*100) < threshold else np.nan})

# Find dates with missing data
missing_dates = ds_daily['TS'][ds_daily['TS'].isnull()].index
print("Dates with missing data:\n",ds_daily.loc[missing_dates])

# Check if the time series is continuous
if pd.date_range(start=ds_daily.index.min(), end=ds_daily.index.max(), freq='D').difference(ds_daily.index).empty:
    print('The time series is continuous \n')
else:
    print('The time series is not continuous \n')
    
# Plot the temperature time series
ds_daily['TS'].plot()


# Instead of an artificial time series for the surface temperature, we now use the observed temperature *TS*.

# In[ ]:


#-----------------------------
# Run the time-dependent heat
# equation
#-----------------------------
D  = 10   # Depth of the domain
dz = 0.5  # Spacing between grid points

# Create a short subset for better visualization
df_hourly_subset = ds_daily.loc['2010':'2014']

# Integrate the heat equation 
T, D, dz = hu.heat_equation_daily(df_hourly_subset['TS'], D, dz, alpha=1.2e-6)

#-----------------------------
# Plot the results
#-----------------------------
# Generate the plotting grid
# y-axis values
y_values = np.arange(-D,0,dz)
# then the x-axis values
x_values = np.arange(len(df_hourly_subset.index))
# Generate the 2D plotting grid
X, Y = np.meshgrid(x_values, y_values)

# Create the filled contour trace
trace = go.Contour(
    x=df_hourly_subset.index,
    y=y_values,
    z=T[::-1],
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
    xaxis=dict(title='Date', tickformat='%d.%m.%Y', tickangle=45, dtick='M3'),
    yaxis=dict(title='Depth [m]'),
    width=1000, height=500,
)

# Create the figure
fig = go.Figure(data=[trace], layout=layout)

# Show the plot
fig.show()

