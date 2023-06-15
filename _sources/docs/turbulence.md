(turbulence)=
# Turbulence and Energy Fluxes 

::::{grid}
:::{grid-item}
:columns: 4
````{card} 
:link: nb_radiation
:link-type: ref
<center><b>Radiation and ground heat flux</b></center>
^^^
```{figure} ./figures/mess.png
:width: 100px
:name: turbulence_icon
```
+++
Explore notebook &rarr;
````
:::

:::{grid-item}
:columns: 4
````{card} 
:link: nb_seb
:link-type: ref
<center><b>Turbulence and surface energy balance</b></center>
^^^
```{figure} ./figures/mess.png
:width: 100px
:name: turbulence_icon
```
+++
Explore notebook &rarr;
````
:::
::::

{Download}`Here <./presentations/Lecture_02.pdf>` you will find the
corresponding lecture slides.


```{admonition} Exercises 
:class: attention 

**Exercise 1**: Given the the following measurements of specific humidity q
[g/kg] and potential temperature θ [K] at two heigths, determine the 
turbulent heat fluxes for each timestep. (Hint: Aussume neutral stability, and
$K_M = 3~m^2s^{−1}$). Plot $Q_H$ and $Q_E$ versus time. Comment on the diurnal
variation of the turbulent heat fluxes. What might be the underlying landcover
in this study area?

| Time      | $\theta_1$ (1 m) | $\theta_2$ (2 m) | $q_1$ (1 m) | $q_2$ (2 m) |
| :-----------: | :-----------: | :-----------: | :-----------: | :-----------: |
| 00:00 h | 278.58 | 278.54 | 3.113 | 3.068 | 
| 06:00 h | 280.12 | 280.17 | 3.051 | 3.065 |
| 12:00 h | 284.28 | 284.35 | 2.985 | 2.996 |
| 18:00 h | 280.02 | 280.01 | 2.675 | 2.628 | 

**Exercise 2**: Assume that the mean vertical eddy moisture flux is $7.2\cdot10^{−4}$
[kg kg-1 m s-1]. What is the turbulent latent heat flux?

**Exercise 3**: What vertical temperature difference is necessary across the
microlayer (bottom 1 mm of the atmosphere) to conduct 300 $W m^{−2}$ of heat
flux?

**Exercise 4**: Find the effective surface heat flux over a forest at neutral
stability when the wind speed is 5 $m s^{−1}$ at a height of 10 m, the surface
temperature is 25ºC, and the air temperature at 10 m is 20ºC. Use an
appropriate method to estimate this flux. Discuss the results.

**Exercise 5**: Given the following measurements from an instrumented tower,
find the sensible and latent heat fluxes. Assume a net radiation of 500 $W m^{−2}$
and a ground flux of 30 $W m^{−2}$.

| z (m) | T (ºC) | r (g/kg) |
| :-----: | :-----: | :------: |
| 10 | 15 | 8 |
| 2  | 18 | 10 |

**Exercise 6**: If the sensible heat flux is 300 $W m^{-2}$ and the latent heat
flux is 100 $W m^{-2}$, what is the Bowen-ratio? What is the likely surface
type?

**Exercise 7**: Find the drag coefficient (bulk coefficient) in statically neutral conditions to
be used with surface winds of 5 $m/s$ at 10 $m$ height over (a) villages, and (b)
grassland. Also, find the friction velocity and surface stress. Discuss the
results.

**Exercise 8**: On an overcast day, a wind speed of 6 $m/s$ is measured with an
anemometer located 10 $m$ over ground within a corn field. What is the wind
speed at 25 $m$ height?

**Exercise 9**: Wind speed is measured at two different heights. Find the
friction velocity using an analytical approach (logarithmic wind law).

| z (m) | u (m/s) |
| :-----: | :-----: |
| 2 | 2 |
|10 | 3.15 |

**Exercise 10**: Given $K_H=5~m^2 s^{-1}$ for turbulence within a stable
background environment, where the local lapse rateis $\partial\theta/\partial
z=0.01~K/m$. Find the kinetic heat flux $\overline{w'\theta'}$.

```
