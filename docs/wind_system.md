(wind)=

::::{grid}
:::{grid-item}
:columns: 4
````{card} 
:link: nb_wind_freq
:link-type: ref
<center><b>Wind frequency</b></center>
^^^
```{figure} ./figures/wind.png
:width: 100px
:name: wind_icon
```
+++
Explore notebook &rarr;
````
:::

:::{grid-item}
:columns: 4
````{card} 
:link: nb_opm
:link-type: ref
<center><b>Mountain waves and orographic precipitation</b></center>
^^^
```{figure} ./figures/wind.png
:width: 100px
:name: wind_icon
```
+++
Explore notebook &rarr;
````
:::
::::

{Download}`Here <https://box.hu-berlin.de/f/adcdd54897094c7a8bc3/>` you will find the
corresponding lecture slides.

```{admonition} Exercises 
:class: attention 

**Exercise 1**: Given a mean wind speed of  $M_0= 5 m~s^{-1}$ and $\alpha=2$,
find the probability that the wind speed will be between 5.5 and 6.5
$m~sˆ{-1}$?
 
**Exercise 2**: Weather station data in CSV format (the file can be downloaded
{Download}`here
<https://raw.githubusercontent.com/sauterto/clim_env_hydro/main/docs/nb/data/weather_station_data_v2.csv>`)
Download weather station data from a high altitude station in Nepal here. In
the csv file you will find, among other things, the wind speed (WS). Calculate
the probabilities of different wind classes, e.g. 1 m/s, 2 m/s ... and fit a
Weibull distribution to the data. Calculate the return period 

**Exercise 3**: Find the steady-state updraft speed in the middle of (a) a
thermal in a boundary layer that is 1 km thick; and (b) a thunderstorm in a 11
km thick troposphere. The virtual temperature excess is 2ºC for the thermal and
5ºC for the thunderstorm, and $|g|/ \overline{T_v} = 0.0333~m \cdot s^{-2}
\cdot K^{-1}$. 

**Exercise 4**: Find the equilibrium updraft speed ($m~s^{-1}$) of a thermal
in a 2 im boundary layer with environmental temperature 15ºC and a thermal
temperture of 19.5 ºC.

**Exercise 5**: Winds of 10 $m~s^{-1}$ are flowing in a valley of 10 km width.
Further downstream, teh valley narrows to the width of 2.5 km. Find the wind
speed in the constriction, assuming constant flow depth.

**Exercise 6**: Assume $g/T_v=0.0333~m \cdot s^{-2} \cdot K^{-1}$. For a
two-layer atmospheric system flowing through a short gap, find the maximum
expected gap wind speed. Flow depth is 300 m, and the virtual potential
temperature difference is 5.5 K.

**Exercise 7**: Anabatic flow is 5ºC warmer than the ambient environment of
15ºC. Find the horizontal and along-slope pressure-gradient forces/mass, for a
30º slope. Furthermore, suppose a steady-state is reached where the two forces
are buoyance and drag. Find the anabatic wind speed, assuming an anabatic flow
depth of 50 m and a drag coefficient of 0.05.

**Exercise 8**: Air adjacent to a 10º slope averages 10ºC cooler over its 20 m
depth than the surrounding air of virtual temperature 10ºC. Find and plot the
wind speed vs. downslope distance, and the equilibrium speed. $C_D=0.005$.
Comment on the differences between the equilibirum speed and the speed derived
from the downslope distance.

**Exercise 9**: Marine-air of thickness 500 m and virtual temperature 16ºC is
advancing over land. The displaced continental-air virtual temperature is 20ºC.
Find the sea-breeze front speed, and the sea-breeze wind speed. 

**Exercise 10**: Assuming calm synoptic conditions (i.e, no large-scale winds
that oppose or enhance the sea-breeze), what maximum distance inland would a
sea-breeze propagate? Use data from the previous exercise, for a latitude of
45ºN. What happens if we are near 30º?

**Exercise 11**: Cold winter air of virtual potential temperature -5ºC and depth
200 m flows through an irregular mountain pass. The air above has virtual
potential temperature 10ºC. Find the maximum likely wind speed through the
short gap.

**Exercise 12**: Find and plot the path of air over a mountain, given:
$z_1=500~m$, $M=30~m~s^{-1}$, $b=3$, $\Delta T/\Delta z=-0.005~K~m^{-1}$,
$T=10ºC$, and $T_d=8ºC$ [Hint: you need to calculate the wavelenght first].
Indicate which waves have lenticular clouds [Hint: the liquid condensation
level can be approcimated with $z_{LCL}=a \cdot (T-T_d)=(125~m~ºC^{-1})\cdot
(T-T_d)$].

**Exercise 13**: For a mountain of width 25 km, find the Froude number. Assume
$g/T_v=0.0333~m \cdot s^{-2} \cdot K^{-1}$, $M=2~m \cdot s^{-1}$, and $\Delta
T/\Delta z=5~ºC \cdot km^{-1}$. Draw a sketch of the type of mountain waves
that are likely for this Froude number.

**Exercise 14**: List and explain commonalities among the equations that
describe the various thermally-driven local flows.

**Exercise 15**: What factors might affect rise rateof the thermals, in
addition to the ones already mentioned?

**Exercise 16**: What factors control the shape of the katabatic wind profile?

**Exercise 17**: What happens to a natural wavelength of air for statically
unstable conditions?

**Exercise 18**: Comment on the differences and similarities of the two
mechanisms for createing Foehn winds.

**Exercise 19**: If air goes over a mountain but there is no precipitation,
would there be a Foehn wind?

**Exercise 20**: Suppose that katabatic winds flow into a bowl-shaped
depression instead of a valley. Descrivbe how the airflow would evolve during
the night.

**Exercise 21**: If warm air was not less dense that cold, could sea-breezes
form? Explain.
```
