(test)=
# Testing the learning environment 

Before we get started, we test the learning environment and the most important
packages needed to run the notebooks. This is not so much a continuous coherent
exercise as individual examples based on the different packages.This exercise
is neither an introduction to Python nor extensive tutorials for the individual
packages. I advise you, if you have little or no experience with the packages,
to work through the relevant tutorial on the websites. All packages offer very
good and extensive tutorials. Most of the functions presented here have been
taken from these websites.

::::{grid}

:::{grid-item}
:columns: 4
````{card} 
:link: nb_testing_pandas
:link-type: ref
<center><b>Getting started with pandas</b></center>
^^^
```{figure} ./figures/pandas.svg
:height: 50px
:name: xarray_icon
```
+++
Explore notebook &rarr;
````
:::

:::{grid-item}
:columns: 4
````{card} 
:link: nb_testing_xarray
:link-type: ref
<center><b>Getting started with xarray</b></center>
^^^
```{figure} ./figures/xarray.png
:height: 50px
:name: xarray_icon
```
+++
Explore notebook &rarr;
````
:::

:::{grid-item}
:columns: 4
````{card} 
:link: nb_testing_metpy
:link-type: ref
<center><b>Getting started with MetPy</b></center>
^^^
```{figure} ./figures/metpy.png
:height: 50px
:name: metpy_icon
```
+++
Explore notebook &rarr;
````
:::

```{admonition} Homework Assignment  
:class: attention 
The Earth's water cycle is expected to change significantly as the climate
warms, affecting societies, economies and ecosystems around the world. Net
water flux to the surface - precipitation minus evaporation over the ocean or
precipitation minus evaporation over land (P - E) - is a key aspect of the
hydrological cycle. On the continents, P - E determines the sum of surface and
subsurface runoff. In this exercise we look at the seasonal characteristics of
(P-E) over Europe and examine the development over the last decade for Berlin. 

1. For the analysis we use ERA5 data from the European Centre for Medium-Range
Weather Forecast (ECMWF). Download from the [ECMWF
website](https://cds.climate.copernicus.eu/cdsapp#/dataset/reanalysis-era5-land-monthly-means?tab=overview)
the monthly averaged ERA5 reanalysis data of total precipitation, total evaporation and runoff for
the period 2013-2022. The study area ranges from 27-72ºN and from 22ºW to 27ºE.
[Hint: Search for 'ERA5-Land monthly data from 1950 to present'. You need to create an account to download the data.]

2. Open the file in Jupyter-Notebook and look at the structure of the file. Find out the variable names. 

3. Use MetPy to parse the data set and add a Coordinate Reference System (CRS).
   Convert the variables to a unit-aware type. [Note: The conversion of the
variable evaporation must be done manually by multiplying the variable by
units['m'].

4. Plot the variables for selected months, e.g. 2022-01-01. In which units are the
variables given? What is striking about the fields?

5. It is often useful to look at the seasonal fields. Calculate the weighted
seasonal mean of the fields and plot the result. What statements can you make
about the mountain regions, and especially about the Alps?

6. To better identify the dry regions, calculate P-E. Reconsider your statements
about the mountain regions. What statements can you make about Germany?

7. Now extract the time series of Berlin (52.52N, 13.4050E) and convert the data
into a Pandas data frame. Plot the time series (precipitation, evaporation and
runoff) for the entire period (2013-2022). Look at the time series P-E. What
statements can you make in relation to the recent drought events in
Brandenburg?


```
