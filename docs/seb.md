(seb_header)=
# Surface Energy Balance model 

### Python Notebooks
* Surface Energy Balance model
  * [Exercises: Surface Energy Balance](seb:exercise)
  * [Solutions: Surface Energy Balance](seb:solution)


### Exercises:


```{figure} ./pics/SEB.png
:width: 700px
:name: SEB_fig
```

>**Task 1**: Develop a simple SEB model. The turbulent flows are to be
>parameterised using a simple bulk approach. Write a function that takes the
>following arguments: surface temperature, air temperature, relative humidity,
>albedo, global radiation, atmospheric pressure, air density, wind speed,
>altitude measured and roughness length. The function should return the
>short-wave radiation balance and the two turbulent energy fluxes. 

>**Task 2**:Now we need to optimize for the surface temperature. Therefore, we
>need to write a optimization function. In our case the sum of all energy
>fluxes should be zero. In this case, the SEB only depends on the surface
>temperature. So we have to find the surface temperature which fulfils the
>condition SEB(T0)=Q0+H0+E0=0. Once the optimization function is written, we
>use the minimize function from the scipy module to find the temperature values.
