#!/usr/bin/env python
# coding: utf-8

# (testing_pandas:exercise)=
# # Using pandas
# 
# Before we get started, we test the learning environment and the most important packages needed to run the notebooks. This is not so much a continuous coherent exercise as individual examples based on the different packages.This exercise is neither an introduction to Python nor extensive tutorials for the individual packages. I advise you, if you have little or no experience with the packages, to work through the relevant tutorial on the websites. All packages offer very good and extensive tutorials. Most of the functions presented here have been taken from these websites.

# <div class="alert alert-block alert-success">
# <b>Learning objectives:</b><br>
# <ul>
#  <li>Getting to know the learning environment</li>
#  <li>Testing the pandas packages</li>
#  <li>Very brief overview of the function of the package</li>
# </ul>  
# </div>

# <div class="alert alert-block alert-info">
# <b>How to proceed:</b><br>
# <ul>
#  <li>Testing pandas</li>
# </ul>  
# </div>

# ## Getting started
# 
# Start using pandas. To load the pandas package and start working with it, import the package. The community agreed alias for pandas is pd. 
# 

# In[ ]:


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

# In[ ]:


# Read the data into a DataFrame
df = pd.read_csv("../data/aws_valley_data_10min.csv", header=1, index_col='TIMESTAMP')


# and have a look at the dataframe

# In[ ]:


# A simple way to plot the DataFrame
df.head()


# We can select a Series from the DataFrame with

# In[ ]:


# Retrieve the air temperature series from the DataFrame
df['AirTC_1']


# do some calculations

# In[ ]:


# Get the maximum of the air temperature series
df['AirTC_1'].max()


# As illustrated by the max() method, you can do things with a DataFrame or Series. pandas provides a lot of functionalities, each of them a method you can apply to a DataFrame or Series. As methods are functions, do not forget to use parentheses ().

# You can also get some basic statistics of the data with

# In[ ]:


df.describe()


# The describe() method provides a quick overview of the numerical data in a DataFrame. Textual data is not taken into account by the describe() method.

# You can simply select specific columns from a DataFrame with

# In[ ]:


# That's how you select the AirTC_1 and RH_1 columns from the df DataFrame
df_subset = df[["AirTC_1","RH_1"]]

# Plot the header (first 5 rows)
df_subset.head()


# The shape of the DataFrame can be accessed with

# In[ ]:


# Access the shape attribute. Please note, do not use parentheses for attributes. 
df_subset.shape


# Often you need to filter specific rows from the DataFrame, e.g.

# <img align="center" valign='top' src="images/03_subset_rows.svg" width=700 >

# With the following command you can simply select all rows with temperatures above 5ºC

# In[ ]:


# Select all rows with temerature greather than 5 degrees celsius
T_subset = df_subset[df_subset["AirTC_1"] > 5.0]

# Plot the header rows
T_subset.head()


# It is possible to combine multiple conditional statements, each condition must be surrounded by parentheses (). Moreover, you can not use or/and but need to use the or operator | and the and operator &. Here is an example

# In[ ]:


# Select all rows with temerature greather than 5 degrees celsius and a relative humidity above 70%
T_RH_subset = df_subset[(df_subset["AirTC_1"] > 5.0) & (df_subset["RH_1"] > 70.0)]

# Plot the header rows
T_RH_subset.head()


# Often you want to create plots from the data.

# <img align="center" valign='top' src="images/04_plot_overview.svg" width=700 >

# To make use of the plotting function you need to load the matplotlib package

# In[ ]:


# Import matplotlib
import matplotlib.pyplot as plt


# You can quickly check the data visually

# In[ ]:


# Plot the temperature time series
df["AirTC_1"].plot()

# Rotate the x-labels for better readability
plt.xticks(rotation=30);


# Or create horizontally stacked plots, add two time series in one plot etc. 

# In[ ]:


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

# In[ ]:


df[["AirTC_1","RH_1","H_Flux"]].plot.box(figsize=(10,5))


# And a simple way to plot all variables in a DataFrame

# In[ ]:


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
