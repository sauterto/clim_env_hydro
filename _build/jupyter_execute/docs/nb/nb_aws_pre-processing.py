#!/usr/bin/env python
# coding: utf-8

# (nb_aws)=
# # Weather station observations
# 
# In this exercise, we will use **pandas**, **xarray**, **MetPy** and **plotly** to analyze weather station data from a csv file. We will perform data cleaning, manipulation, and visualization to gain insights into the data and explore its characteristics.
# 

# <div class="alert alert-block alert-success">
# <b>Learning objectives:</b><br>
# <ul>
#  <li>Get familiar with weather station data</li>
#  <li>Quality assessment of measurement time series</li>
#  <li>Data resampling</li>
#  <li>Visualisation with interactive plots</li>
# </ul>  
# </div>

# <div class="alert alert-block alert-info">
# <b>Prerequisites</b><br>
# <ul>
# <li>Basic knowledge of Python, Jupyter Notebooks, and data analysis</li>
# <li>Familiarity with MetPy, Pandas, and Xarray</li>
# <li>A csv file containing weather station data (can be downloaded <a href="https://github.com/sauterto/clim_env_hydro/blob/main/docs/nb/data/FLX_CH-Dav_missing.csv" download>here</a>)</li>
# </ul>  
# </div>

# ## Load weather station data
# 
# In this example, the code reads in the 30-min weather stations data from a CSV file using the '**read_csv()**' method from pandas, with the parse_dates and index_col parameters set to True and 0, respectively. This ensures that the first column of the CSV file (the timestamps) is used as the index of the DataFrame, and that the timestamps are converted to datetime objects.

# In[ ]:


import pandas as pd
import numpy as np

# Load CSV file
df = pd.read_csv("https://raw.githubusercontent.com/sauterto/clim_env_hydro/main/docs/nb/data/FLX_CH-Dav_missing.csv", parse_dates=True, index_col=0)

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

# In[ ]:


df.head()


# The DataFrame has missing values with the dummy value -9999. We then use the replace method to replace all occurrences of -9999 with NaN (Not a Number), using the NumPy np.nan constant. Finally, we print the updated DataFrame. The inplace=True argument ensures that the original DataFrame is modified, rather than a copy being created.

# In[ ]:


# replace -9999 with NaN
df.replace(-9999, np.nan, inplace=True)
df


# <div class="alert alert-block alert-warning">
# <b>Hint!</b> Compare the SW_OUT and LW_OUT column with those of the previous table.
# </div>

# To check for missing values in a Pandas DataFrame, you can use the **isna()** or **isnull()** method, which returns a Boolean DataFrame of the same shape indicating which cells contain missing values.

# In[ ]:


# check for missing values
print(df.isna())


# You can also use the isna() or isnull() method along with the **sum()** method to count the number of missing values in each column:

# In[ ]:


# count the number of missing values in each column
print(df.isna().sum())


# <div class="alert alert-block alert-warning">
# <b>Broaden Knowledge & Comprehension</b></br> Find out how to delete all rows containing NaNs with pandas. Note: Check the pandas webpage.
# </div>

# We can also check whether whole periods (missing timestamps) are missing, by generating a date range using the '**date_range()**' method from pandas, with the start and end parameters set to the minimum and maximum timestamps in the DataFrame, respectively, and the freq parameter set to the expected frequency of the time series (e.g. '30T' for every 30 minutes).
# 
# The "**difference()**' method is then used to compare the date range to the DataFrame index. If there are any missing periods in the time series, then the resulting set will be non-empty, and the code will print 'The time series is not continuous'. Otherwise, the time series is continuous and the code will print 'The time series is continuous'.

# In[ ]:


# Check for missing periods
if pd.date_range(start=df.index.min(), end=df.index.max(), freq='30T').difference(df.index).empty:
    print('The time series is continuous \n')
else:
    print('The time series is not continuous \n')
    print(('These dates are missing: \n {0}').format(pd.date_range(start=df.index.min(), end=df.index.max(), 
                                                                 freq='30T').difference(df.index)))


# Calculate some statistics and check if ranges are reasonable

# In[ ]:


# 95th-Quantile (extremes)
df.quantile(q=0.95)


# In[ ]:


# Standard deviation
df.std()


# In[ ]:


# Mean values
df.mean()


# <div class="alert alert-block alert-warning">
# <b>Broaden Knowledge & Comprehension</b></br> Are these values reasonable?
# </div>

# ## Resample data
# 
# The '**resample()**' method is then used to resample the data to hourly frequency, with different aggregation methods specified for each column using the '**agg()**' method. The Temperature column is aggregated using the mean() method, the Relative Humidity column is aggregated using the min() method, the Wind Speed column is aggregated using the max() method, the Wind Direction column is aggregated using the last() method, and the Radiation fluxs are aggregated using the mean() method. This effectively aggregates the data to hourly intervals using different aggregation methods for each column.

# In[ ]:


# Resample the data to hourly frequency with different aggregation methods for each column
resampled_hourly = df.resample('1H').agg({'t2m': 'mean', 'RH': 'min',
                                  'WS': 'max', 'WD': 'last', 'NETRAD': 'mean',
                                 'precip':'sum', 'SW_IN':"mean", 'SW_OUT':"mean",
                                 'LW_IN':"mean", 'LW_OUT':"mean", "H":"mean", 'LE':"mean",
                                  'QG':"mean", 'TS':"mean"})

# Find dates with missing data
missing_dates = resampled_hourly['t2m'][resampled_hourly['t2m'].isnull()].index
print("Dates with missing data:\n",resampled_hourly.loc[missing_dates])

# Check if the time series is continuous (having missing dates)
if pd.date_range(start=df.index.min(), end=df.index.max(), freq='H').difference(resampled_hourly.index).empty:
    print('The time series is continuous \n')
else:
    print('The time series is not continuous \n')


# <div class="alert alert-block alert-warning">
# <b>Broaden Knowledge & Comprehension</b></br> How do you modify the code to get annual means? Note: See the information on the Pandas page about the resample method, and check <a href="https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases">here</a> for a list of frequency aliases.
# 
# </div>

# If there are long periods without data, the resampling process will still produce hourly intervals with NaNs in place of missing values. The length of the gaps between valid data points will be reflected in the number of consecutive NaN values in the resampled DataFrame.
# 
# You can also simply determine the minimum number of elements that must be included in the resampling. To resample a DataFrame and return NaN values for columns that do not have sufficient number of elements in the resampling set, you can use the '**resample()**' method in combination with''**agg()**' method and a custom aggregation function that checks for the number of elements in each column and returns NaN if the count is less than a specific threshold. Here's an example:

# In[ ]:


# resample dataframe to monthly mean values with different aggregation for each column and return NaN for 
# columns with insufficient valid elements
threshold =  10 # What is the maximum percentage of NaNs that may be included in the averaging? Here, we set the threshold to 10%

# This is a pythonic way in solving this problem
resampled_thres = resampled_hourly.resample('1M').agg({'t2m': lambda x: x.dropna().mean() if (((x.isna().sum())/len(x))*100) < threshold else np.nan,
                                    'RH': lambda x: x.dropna().mean() if (((x.isna().sum())/len(x))*100) < threshold else np.nan})

# Find dates with missing data
missing_dates = resampled_thres['t2m'][resampled_thres['t2m'].isnull()].index
print("Dates with missing data:\n",resampled_thres.loc[missing_dates])

# Check if the time series is continuous
if pd.date_range(start=df.index.min(), end=df.index.max(), freq='H').difference(resampled_hourly.index).empty:
    print('The time series is continuous \n')
else:
    print('The time series is not continuous \n')
    
# Plot the temperature time series
resampled_thres['t2m'].plot()


# <div class="alert alert-block alert-warning">
# <b>Tip!</b> Change the threshold (percetage) and watch how the time series changes.
# </div>

# Here is another example. We can calculate the seasonal mean by resampling the data using the resample() method, and taking the mean of each season:

# In[ ]:


# Resample to the seasonal mean starting from December
seasonal_mean = df.resample('QS-DEC').agg({'t2m': 'mean', 'RH': 'min',
                                  'WS': 'max', 'WD': 'last', 'NETRAD': 'mean',
                                 'precip':'sum', 'SW_IN':"mean", 'SW_OUT':"mean",
                                 'LW_IN':"mean", 'LW_OUT':"mean", "H":"mean", 'LE':"mean",
                                  'QG':"mean", 'TS':"mean"})

# Find out which year had the hottest season
hottest_season_year = seasonal_mean['t2m'].idxmax().year

# Print the hottest year
print(('The hottest year was in {0}').format(hottest_season_year))


# To find the hottest seasons (indices) above a certain quantile using pandas, you can use the quantile() method to calculate the value of the quantile, and then use boolean indexing to filter the DataFrame based on the values above that quantile.

# In[ ]:


# Calculate the 95th percentile
quantile_95 = seasonal_mean['t2m'].quantile(0.95)

# Filter the DataFrame based on values above the 95th percentile
indices_above_quantile = seasonal_mean[seasonal_mean['t2m'] > quantile_95].index

# Print the hottest years above the 95% quantile
print('The hottest years were in: \b')
print(pd.Series(indices_above_quantile.format()))


# <div class="alert alert-block alert-warning">
# <b>Broaden Knowledge & Comprehension</b></br> How do you modify the code to get annual means? Note: See the information on the Pandas page about the resample method, and check <a href="https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases">here</a> for a list of frequency aliases. Try to identify the coldest year.
# </div>

# ## Visualize temperature and relative humdity
# 
# Finally, we can visualize the data to identify patterns and trends in the weather. We use the '**plotly**' library to create interactive plots. In this code, we use '**make_subplots()**'
#  to create two panels in a single figure, with a shared x-axis. We then added the temperature and precipitation data as dashed lines in their respective panels using '**go.Scatter()**'
# .
# 
# 

# In[ ]:


# Import the plotly library
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# Creating the plot with two rows and one column. The plots share the same x-axis so that only labels 
# for the lower plots are shown
fig = make_subplots(rows=2, cols=1, shared_xaxes=True)

# Adding the temperature data as a dashed line in the top panel
fig.add_trace(go.Scatter(x=resampled_thres.index, y=resampled_thres['t2m'], line=dict(color='royalblue', dash='solid'), name='Temperature (°C)'),
                         row=1, col=1)

# Adding the precipitation data as a dashed line in the bottom panel
fig.add_trace(go.Scatter(x=resampled_thres.index, y=resampled_thres['RH'], line=dict(color='green', dash='solid'), name='Precipitation (mm)'),
                         row=2, col=1)

# Adjusting the layout
fig.update_layout(title='Temperature and Precipitation', plot_bgcolor='white', width=800, height=600,
                  yaxis=dict(title='Temperature (°C)', showgrid=True, gridcolor='lightgray', gridwidth=1),
                  yaxis2=dict(title='Precipitation (mm)', showgrid=True, gridcolor='lightgray', gridwidth=1),
                  xaxis=dict(title='', tickformat='%d.%m.%Y', showgrid=True, gridcolor='lightgray', gridwidth=1),
                  xaxis2=dict(title='Date', tickformat='%d.%m.%Y', showgrid=True, gridcolor='lightgray', gridwidth=1))


# Adjusting the axes
fig.update_xaxes(nticks=10, row=1, col=1)
fig.update_yaxes(nticks=10, row=1, col=1)
fig.update_xaxes(nticks=10, row=2, col=1)
fig.update_yaxes(nticks=10, row=2, col=1)

# Showing the plot
fig.show()


# ## Visualize precipitation data
# 
# 

# In[ ]:


import plotly.express as px

# Resample the data to hourly frequency with different aggregation methods for each column
resampled_monthly = df.resample('1M').agg({'t2m': 'mean', 'RH': 'min',
                                  'WS': 'max', 'WD': 'last', 'NETRAD': 'mean',
                                 'precip':'sum', 'SW_IN':"mean", 'SW_OUT':"mean",
                                 'LW_IN':"mean", 'LW_OUT':"mean", "H":"mean", 'LE':"mean",
                                  'QG':"mean", 'TS':"mean"})

# Create a bar plot with Plotly Express
fig = px.bar(resampled_monthly, x=resampled_monthly.index, y='precip', color='precip', title='Monthly Precipitation')

# Add axis labels and title
fig.update_layout(xaxis_title='Month', yaxis_title='Precipitation (mm)', title_x=0.5,
                  plot_bgcolor='white', width=800, height=400,)

# Show the plot
fig.show()


# We can also plot the distribution of the monthly precipitation. To do this, we estimate the probability density function with a kernel-density estimate using Gaussian kernels '**gaussian_kde()**'. Kernel density estimation is a way to estimate the probability density function (PDF) of a random variable in a non-parametric way.

# In[ ]:


import numpy as np
import scipy.stats as stats
import plotly.express as px

# estimate PDF with KDE
kde = stats.gaussian_kde(resampled_monthly['precip'])

# create a grid of x values for plotting
x_vals = np.linspace(0, resampled_monthly['precip'].max(), num=50)

# evaluate the PDF at the x values
pdf_vals = kde.evaluate(x_vals)

# plot the PDF with Plotly
fig = px.line(x=x_vals, y=pdf_vals)
fig.show()


# ## Quality assessment of wind data
# 
# We remove rows where the wind speed is greater than 100, since that value is likely an outlier. Finally, we remove any rows where the wind direction is not between 0 and 360 degrees, or where the wind direction is not a number using boolean indexing. You can modify this code to perform additional quality assessment checks, depending on your specific requirements.

# In[ ]:


# Drop rows with missing wind data
df.dropna(subset=["WS", "WD"], inplace=True)

# Remove rows where wind speed is 0
df = df[df["WS"] > 0]

# Remove rows where wind speed is greater than 100
df = df[df["WS"] <= 100]

# Remove rows where wind direction is not between 0 and 360 degrees
df = df[(df["WS"] >= 0) & (df["WS"] <= 360)]

# Remove rows where wind direction is not a number
df = df[np.isfinite(df["WD"])]


# ## Create a Windrose
# 
# In this example, we use the **rosely** library to create a WindRose object to create the windrose chart. 
# 

# In[ ]:


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


# <div class="alert alert-block alert-warning">
# <b>Broaden Knowledge & Comprehension</b></br> Plot the statistics for other sectors.
# </div>

# ## Calculate some indices
# 
# In this example, we load the weather data from a CSV file and convert the temperature from Fahrenheit to Celsius. We then calculate the wind direction and speed, dew point temperature, heat index, and wind chill using MetPy functions. We also create a daily resampled DataFrame and plot the daily mean temperature and a wind rose.

# In[ ]:


import metpy.calc as mpcalc
from metpy.units import units

# Convert the temperature from Fahrenheit to Celsius
resampled_monthly['t2m'] = ((resampled_monthly['t2m'] - 32) * 5/9).round(2)

# Calculate the wind direction from the U and V components
u, v = mpcalc.wind_components(resampled_monthly['WS'].values * units('m/s'), 
                              resampled_monthly['WD'].values * units.deg)
wind_direction = mpcalc.wind_direction(u, v).to('deg').magnitude
resampled_monthly['WD'] = wind_direction.round(2)

# Calculate the wind speed in knots
resampled_monthly['WS kts'] = resampled_monthly['WS'].values * units('m/s').to('knots')

# Calculate the relative humidity
resampled_monthly['Dew Point'] = mpcalc.dewpoint_from_relative_humidity(resampled_monthly['t2m'].values * units.degC, resampled_monthly['RH'].values * units.percent)

# Calculate the heat index
resampled_monthly['Heat Index'] = mpcalc.heat_index(resampled_monthly['t2m'].values * units.degC, 
                                       resampled_monthly['RH'].values * units.percent).to('degC').round(2)

# Calculate the wind chill
resampled_monthly['Wind Chill'] = mpcalc.windchill(resampled_monthly['t2m'].values * units.degC, 
                                      resampled_monthly['WS'].values * units('m/s')).to('degC').round(2)


# <div class="alert alert-block alert-warning">
# <b>Broaden Knowledge & Comprehension</b></br>What other indices can we calculate with MetPy?
# </div>

# Let's create a plot. Here, we plot the difference between the temperature and dew point temperture. This is another measure to identify wet/dry months.

# In[ ]:


# Creating the plot with two rows and one column. The plots share the same x-axis so that only labels 
# for the lower plots are shown
fig = make_subplots(rows=1, cols=1)

data = resampled_monthly

# Adding the temperature data as a dashed line in the top panel
fig.add_trace(go.Scatter(x=data.index, y=data['t2m']-data['Dew Point'], line=dict(color='royalblue', dash='solid'), name='Temperature (°C)'))

# Adjusting the layout
fig.update_layout(title='Difference Plot between temperature and dew point', plot_bgcolor='white', width=1200, height=800,
                  yaxis=dict(title='Temperature (°C)', showgrid=True, gridcolor='lightgray', gridwidth=1),
                  yaxis2=dict(title='Precipitation (mm)', showgrid=True, gridcolor='lightgray', gridwidth=1),
                  xaxis=dict(title='', tickformat='%d.%m.%Y', showgrid=True, gridcolor='lightgray', gridwidth=1),
                  xaxis2=dict(title='Date', tickformat='%d.%m.%Y', showgrid=True, gridcolor='lightgray', gridwidth=1))


# Adjusting the axes
fig.update_xaxes(nticks=10, row=1, col=1)
fig.update_yaxes(nticks=10, row=1, col=1)
fig.update_xaxes(nticks=10, row=2, col=1)
fig.update_yaxes(nticks=10, row=2, col=1)

# Showing the plot
fig.show()


# <div class="alert alert-block alert-warning">
# <b>Broaden Knowledge & Comprehension</b></br> What are the driest months?
# </div>

# In[ ]:




