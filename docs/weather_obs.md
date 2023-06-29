(aws)=
# Weather observations 

::::{grid}
:::{grid-item}
:columns: 4
````{card} 
:link: nb_aws
:link-type: ref
<center><b>Weather Station</b></center>
^^^
```{figure} ./figures/aws.png
:width: 100px
:name: glacier_icon
```
<center>This notebook deals with the basics of climatology</center>
+++
Explore notebook &rarr;
````
:::
::::

{Download}`Here <https://box.hu-berlin.de/f/f9184480b9a1486b8b74/>` you will find the
corresponding lecture slides.

```{admonition} Homework Assignment  
:class: attention 

**Title**: Weather station data analysis and comparison with ERA5 reanalysis data

**Objective**:

To perform an in-depth quality assessment of weather station data
and compare it with ERA5 reanalysis data using Python and Jupyter notebooks.
The exercise can be solved using the knowledge and skills taught in the
provided exercise notebooks, including data importing, data cleaning and
quality assessment, data visualization, and comparison with ERA5 data. Make
sure to review the notebooks carefully and use the appropriate functions and
libraries to complete the exercise.

**Materials**:

- Weather station data in CSV format (the file can be downloaded {Download}`here <https://raw.githubusercontent.com/sauterto/clim_env_hydro/main/docs/nb/data/weather_station_data_v2.csv>`). The station is located at 81.4721E and
  30.2711N, and measures temperature (T), relative humidity(RH), air pressure
(Pressure), shortwave radiation (Radiation), wind speed (WS), and wind
direction (WD)
- ERA5 reanalysis data in netCDF format (needs to be downloaded from the ECMWF webpage)
- Jupyter notebook with Python 3 kernel
- Python libraries: pandas, xarray, MetPy, plotly, rosely

*Procedure:*

**Part 1: Quality Assessment of Weather Station Data**

1. Import the weather station data into a pandas DataFrame using the read_csv function.
2. Check for missing data, outliers, and unrealistic values. Remove or fill in missing data and outliers as necessary.
    - Are there any outliers? If so, which ones and when do they occur?
    - Are there any missing data? If so, when do they occur and try to fill them?
3. Calculate basic statistics (e.g. mean, standard deviation, minimum, maximum) for each variable in the data set. Analyze the statistics to identify any unusual patterns or trends.
    - Do the statistics meet expectations?
    - What are the monthly, seasonal and annual averages?


**Part 2: Visualization and Analysis of Weather Station Data**

1. Create graphs to visualize the weather station data using matplotlib and/or plotly. Use line graphs for time series data and a windrose plots for wind speed and direction.
2. Analyze the data and answer the following questions:
    - What climate zone is the station located in?
    - What is the wettest/driest and warmest/coldest season?
    - What is the predominant wind direction?
    - Can differences in wind direction be detected during the course of the day?

**Part 3: Comparison of Weather Station Data with ERA5 Reanalysis Data**

1. Download ERA5 reanalysis data for temperature, relative humidity and wind for the same region/location as the weather station data.
2. Import the ERA5 reanalysis data using xarray.
3. Calculate basic statistics for each variable in the ERA5 reanalysis data set.
4. Compare the ERA5 reanalysis data with the weather station data using MetPy, xarray, pandas and rosely. 
    - Correlate the ERA5 data with the station data using the **xarray.corr()** function. Check the xarray webpage for details.
    - How well does the ERA5 reanalysis data represent the location?
    - Which variables are well represented, which less so?
    - Does the agreement improve when monthly or annual averages are compared?

**Assessment**:

 Your work will be assessed on the basis of the accuracy and completeness of
the analysis and the ability to present your results clearly and concisely. It
should be evident that you have a basic understanding of the process of
assessing the quality of weather station data and comparing it with ERA5
reanalysis data.




```

