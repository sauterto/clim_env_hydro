(ebm_header)=
# Exercise: Glacier Winds  

 ```{figure} ./pics/katabatic_shaw.png
:height: 600px
:name: shaw
Figure 1: A schematic of the potential wind flow and interplay of local effects
on Tsanteleina Glacier in the Italian Alps. The diagram represents wind
modelling, measured data and observations from the field. (1) The interactions
of down-glacier katabatic winds (blue) and up-valley winds/local heat sources
(red); (2) the potential heat emitted from the warm valley surroundings (pink)
and; (3) localised surface depressions representing glacier 'cold spots' during
calm, high pressure conditions. Arrows correspond to synoptic westerlies
(purple), southerly airflow (orange), katabatic winds (blue) and valley winds
(red) [Credit: T Shaw, https://blogs.egu.eu/divisions/cr/2017/01/25/katabatic-winds-a-load-of-hot-or-cold-air/] 
```

Katabatic flow associated with the stable boundary layer (SBL)
often develop above glacier when advection of warm air over the much cooler glacier
surface leads to a strong stratification and downward directed buoyancy flux ({numref}`shaw`).
The permanent cold glacier surface produces a shallow cold air layer above the
ground, which drains down the slopes following the local topography. The
development of persistent shallow (5-100 m) downslope winds above glaciers are
a well known phenomena and commonly referred to as glacier wind. The
characteristic and intensity of the glacier wind is governed by the interplay
of buoyancy, surface friction and entrainment at the upper boundary of the SBL.
Near the surface the wind is usually calm and the wind velocity gradually
increases with height, frequently forming a pronounced low-level jet (LLJ).
Above the LLJ winds gradually changes to geostrophic.

 ```{figure} ./pics/glacier_wind_oerlemans.png
:height: 300px
:name: oerlemans

Observed wind and temperature profiles on 29 July 2007. Profiles are shown for every 3 hours
(UT), but represent 30-min averages. (Source: Oerlemans, 2010)
```


 ```{figure} ./pics/SBL_schematic.png
:height: 400px
:name: ebm_sketch

Boundary layer processes over mountain glaciers. Shown are the wind profile
(U), potential temerature profile (theta), low-level jet (LLJ), downward
directed heat flux (Qh), similarity relationships (local and z-less scaling
regions) and the origin of downburst events.
```
In alpine regions, well developed glacier winds often show a wind maximum in
the lowest 1-10 meters above the surface ({numref}`oerlemans`). Usually the
strongest winds occur during the warmest air temperatures. The observations
imply that there is a correlation between the height and strength of the
katabatic wind - the stronger the jet, the higher the maximum.
Furthermore, the height of the beam shows a dependence on the slope. The
steeper the terrain, the lower the maximum.



### Learning objectives:
* A basic understanding of glacier winds
* Simplified dynamic equations describing katabatic flow
* Steady-state Prandtl model for glacier wind 

### After the exercise you should be able to answer the following questions:

### Problem description:

 ```{figure} ./pics/prandtl_schematic.png
:height: 400px
:name: prandtl

Schematic temperature and wind profiles over glacier slopes.
```

The Navier-Stokes equations describes the motion of fluids. For shallow
steady-state katabatic flow we can simplify these equations by using the Boussinesq
approximation and assuming a hydrostatic equilibrium. Furthermore, we assume
that friction balances the acceleration by buoyancy forcing. Thus, the first-order momentum and heat budgets can be written as

$$
\frac{g \cdot sin(\eta)}{T_0}\theta = \frac{\partial F_u}{\partial z}
$$ (momentum)

$$
-\gamma_{\theta} \cdot sin(\eta) \cdot u = \frac{\partial F_{\theta}}{\partial z}
$$ (heat)

with $g$ the gravitational acceleration, $T_0$ the characteristic temperature, $F_u$ the turbulent momentum flux, $F_{\theta}$ the turbulent heat flux, $z$ the height above the ground, $u$ the wind speed, $\theta$ the potential temperature, and $\eta$ the glacier slope. To close the equation we parametrize the momentum and heat flux with simple K-theory:

$$
F_u = -K_m \frac{du}{dz}, F_{\theta} = -K_h \frac{d\theta}{dz}.
$$ (k_theory)

The two constants $K_h$ and $K_h$ are the eddy diffusivities for momentum and heat. Pluggin these equations into Eq.{eq}`momentum` and {eq}`heat` we obtain:

$$
\frac{g \cdot sin(\eta)}{T_0} \theta + \frac{d}{dz}\left(K_m \frac{du}{dz}\right) = 0.
$$ (momentum_eq)

$$
-\gamma_{\theta} \cdot sin(\eta) \cdot u + \frac{d}{dz}\left(K_h \frac{d\theta}{dz}\right) = 0.
$$ (heat_eq)

To sake of simplicity we also write $s=-sin(\eta) ~ (>0)$. Prandtl (1942) solved these equation to understand thermally induced slope flows. The final equation can be written as:

$$
K_m \frac{d^2 u}{dz^2} - \frac{g \cdot s}{T_0} \theta = 0.
$$ (ode_momentum) 

$$
K_h \frac{d^2 \theta}{dz^2} - \gamma_{\theta} \cdot s \cdot u = 0.
$$ (ode_heat)

This set of equation form a system of homogeneous linear differential equations of fourth order. 
The general solution can be found using a linear combination of the fundamental basis function

$$
u(z) = \sum_{i=1}^{4} a_i e^{\lambda_i z}, \theta(z) = \sum_{i=1}^{4} a_i e^{\lambda_i z}.
$$ (characteristic_function)

The constants and $a_i$ and the the eigenvalue $\lambda_i$ are both omplex. Using the following boundary condition:

$$
u(z=0, z \rightarrow \inf) = 0,
$$ (bc_u)

$$
\theta(z \rightarrow \inf) = 0, \theta(z=0)=C,
$$ (bc_theta)

we find the general solutions.

```{admonition} Analytical Prandtl-Model

The equations that fullfills the conditions are

$$
\theta(z) = C \exp^{-z/\lambda} \cos(z/\lambda)
$$ (sol_theta)

$$
u(z) = C \mu \exp^{-z/\lambda} \sin(z/\lambda)
$$ (sol_u)

with

$$
\lambda=\left(\frac{4 \cdot T_0 \cdot K_m \cdot K_h}{g \cdot s^2 \cdot \gamma_{theta}}\right)^{\frac{1}{4}}
$$ (lambda)

$$
\mu = \left( \frac{g \cdot K_h}{T_0 \cdot K_m \cdot \gamma_{\theta}}\right)^{\frac{1}{2}}
$$ (mu)
```

### Tasks 
1. Implement theaanalytical  Prandtl-Model in Pyton 
2. Run the code for several initial and parameter combination. What general statements can be derived from the simulations?
3. Scale the height with the natural length scale $\lambda$ of the flow, temperature
   with $C$ and wind speed with $\mu C$. Plot the wind and temperature profile
again. At what height is the wind maximum? Derive the height of the wind
maximum from Eq. {eq}`sol_u` . 



```{admonition} Revisit the EBM-Model
So far, the greenhouse effect has been parameterised by tau in the energy
balance model. However, the transmissivity (clouds etc.) fluctuates with the
development of the weather. The simple model has no information about the
dynamics. In order to nevertheless include dynamics, we modify the energy
balance model a little and generate a new tau at each time step. To do this, we
sample the transmission values from a normal distribution with a standard
deviation of 3 percent. 

- Run the energy balance model $T(0)=288 ~ K$, $C_w= 2\cdot10^8 ~ J/(m^2
\cdot K)$, $\alpha=0.3$, and $\tau_{mean}=0.64 (\pm 3\%)$

- Yet, the model does not take into account changes in albedo that result
  from changes in glaciation and land use as a consequence of a changing
climate. Therefore, we are now extending the model with a simple ice/land use
albedo parameterisation. In this parameterisation, the albedo is solely a
function of mean temperature. As a non-linear function we assume a sigmoid function with 

$$
\alpha(T_i) = 0.3 \cdot (1-0.025 \cdot \tanh(1.548 \cdot (T_i-288))).
$$

Carry out the following simulations:
1) Run the energy balance model with four different initial conditions for
$T(0)=288 ~ K$, while fixing the other parameters to $C_w= 2\cdot10^8 ~ J/(m^2
\cdot K)$, $\alpha=0.3$, and $\tau_{mean}=0.64%)$
What can be said about the state of equilibrium?

2) Repeat the previous simulation, but again sample the transmissivity on a
normal distribution with a standard deviation of 3%.  What special feature can
now be observed? What conclusions can be inferred regarding the prediction of
weather and climate?


Execute the script with four different initial conditions for $T(0)= [286, 293, 288.6, 288.9] ~ K$. 

- What are the disadvantages of this simplified model?

```



