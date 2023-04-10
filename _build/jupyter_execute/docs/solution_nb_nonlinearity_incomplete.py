#!/usr/bin/env python
# coding: utf-8

# (nonlinearity:solution_incomplete)=
# ### Von-May function
# 
# **Task 1**: Write a function which solves the Von-May-Equation.
# 
# 
# **Problem description:**
# 
# <blockquote>The starting point for our analysis is the ‘Von-May-Equation’, which is given by <br>
# 
#     
# **\begin{align}    
# y_{t+1} = r \cdot y_{t} \cdot (1-y_{t}),
# \end{align}**
# 
# with  $r$ an pre-defined parameter and $y$ the function value at time $t$ and $t+1$.</blockquote>

# In[58]:


import matplotlib.pyplot as plt

def von_may(y0,r):
    '''
    This function integrates the Von-May Equationn using a then initial condition y0, 
    and the parameter r
    
    y0 :: initial value
    r  :: pre-defined parameter
    
    '''

    # Assignments
    # The variable yi contains always y-value at timestep t. 
    # At the beginning, we assume that the old value 
    # corresponds to the intial value. 
    yi = y0  
    
    # The newly calculated y-values are stored in the result list. 
    # The first value in the list is the initial value for y
    result = [yi]

    # Integrate the Von-May equation over 500 time steps
    for t in range(500):
        # Here, we calculate the new y-values according to the Von-May-Gleichung
        y = r*yi*(1-yi)    
        
        # Store newly calculated values to yi (old value for the next iteration)
        yi = y   
        
        # Append the new y-value to the result list
        result.append(y)   

    # Return the result list
    return(result)



# **Task 2:** Run the code for several initial and parameter combination. What is particularly striking about increasing r-values?
# 
# 
# ```
# y(0)=0.5 and r=2.80 (alternatively, use y(0)=0.9) 
# y(0)=0.5 and r=3.30 (alternatively, use y(0)=0.9) 
# y(0)=0.5 and r=3.95 (alternatively, use y(0)=0.495) 
# y(0)=0.8 and r=2.80 
# 
# ```

# In[59]:


# Run the von_may function and store the result in the variable res
res = von_may(0.5, 3.95)

# Plot the equation
# Initialize the figure
plt.figure(figsize=(20,10))

# Plot the res-List
plt.plot(res);


# **Extend the Von-May function**
# 
# **Task 3:** Extend this Von-May function by generating 20 random r-values and run simulations with them. Sample the values from a normal distribution with mean 3.95 and standard deviation 0.015 (limit the r-values between 0 and 4). Then average over all time series. Plot both the time series, the averaged time series and the histogram of the averaged time series. What do you observe?

# In[62]:


# Import some modules which are used in the function
import random         # Module to generate random number
import numpy as np    # Numpy

def ensemble_may(n, y0, r):
    '''
    The function runs n ensemble members of the Von-May-Equation. The function takes the 
    initial condition y0, the parameter r, and the number of ensemble members n.

    Example: ensemble(50, 0.5, 3.95)

    n    :: number of ensemble members
    y0   :: initial value
    r    :: pre-defined parameter

    '''

    # Assignments   
    # The ensemble members are stored in the result list. 
    # Here, we initialize an empty list
    result = []
    
    # Initialize the random number generator (see random module)
    random.seed()
    
    # Generate n ensemble members (n-loops)
    for ens_member in range(n):
        
        # Here, we limit the randomly generated r-value between 0 and 4
        # We initialise rnd with -999 so that the condition is fulfilled 
        # in the following while loop. The while loop is executed until 
        # the random r parameter is between 0 and 4.
        rnd = -999
        
        # set constraints for the random number 0<rnd<4
        while rnd<=0 or rnd>4:
            # Generate random r-parameter rnd with the mean r and the standard
            # deviation 0.02
            rnd = random.normalvariate(r,0.02);      
                
        # Integrate the von-may equation with a random r-parameter
        result.append(von_may(y0,rnd))
        
    # Return the result
    return(result)



# In[63]:


# Execute the ensemble_may function and store the result in ens
ens = ensemble_may(10, 0.5, 3.93)

# Here, we take the time-mean of each of the ensemble members
# First, the list is converted to a numpy array (np.array(ens)). After 
# that the mean along each row is calculated (np.mean())
ens_mean = np.mean(np.array(ens),axis=0)

# Create two subplot
fig, ax = plt.subplots(1,2, figsize=(15,5))
ax[0].hist(ens_mean, 20);
ax[1].plot(ens_mean);


# **Revisit the EBM-Model**
# 
# Include a dynamic transmissivity in the energy balance model.
# 
# **Task 4:** Run the energy balance model $T(0)=288 ~ K$, $C_w= 2\cdot10^8 ~ J/(m^2 \cdot K)$, $\alpha=0.3$, and $\tau_{mean}=0.608 (\pm 10\%)$

# In[64]:


# Import some modules which are used in the function
import random         # Module to generate random number
import numpy as np    # Numpy

def OLR(T, tau):
    """ Stefan-Boltzmann law """
    sigma = 5.67e-8              # Stefan-Boltzmann constant
    return tau * sigma * T**4    # Return the OLR calculated by the Stefan-Boltzmann law

def ASR(Q, alpha):
    """ Absorbed shortwave radiation """
    return (1-alpha) * Q         # Return the ASR, with the albedo value alpha


def step_forward(Q, T, Cw, alpha, tau, dt):
    # Return the updated T-value (time-dependent EBM)    
    return T + dt / Cw * ( ASR(Q, alpha) - OLR(T, tau) ) 


# In[65]:


def ebm_stochastic(T0, Q=341.3, Cw=10e8, alpha=0.3, tau=0.64, years=100):
    ''' This is a simple Energy Balance Model with a random tranmissivity.'''
  
    # Create result arrays (numpy) filled with zeros
    # Ts stores the temperature values, years the years since the beginning of
    # the simulation
    Ts    = np.zeros(years+1)
    Years = np.zeros(years+1)
    
    # Timestep in seconds (time step is 1 year)
    dt = 60*60*24*365                  # convert days to seconds

    # Initial and boundary conditions
    # Set the first value in the Ts to the initial condition
    Ts[0] = T0 

    # Integration over all years
    for n in range(years):
        # Generate a random tau value with mean value tau and standard-deviation
        # of 10 % its value
        tau_rnd = random.normalvariate(tau,tau*0.1)
        # Store the number of iterations in the Years array
        Years[n+1] = n+1
        # Store the new temperature value in Ts
        Ts[n+1] = step_forward( Q, Ts[n], Cw, alpha, tau_rnd, dt )
        
    # Return both the temperature and year array
    return Years, Ts
        


# In[66]:


# Execute the ebm_stochastic function for 1000 years
yrs, Ts = ebm_stochastic(273, Q=342, Cw=2*10e8, alpha=0.30, tau=0.608, years=1000)

# Plot the results
fig, ax = plt.subplots(1,1,figsize=(20,5))
ax.plot(yrs,Ts);


# **Extend the model with a simple ice/land use albedo parameterisation. (sigmoid version)**
# 
# **Task 5:** In this parameterisation, the albedo is solely a function of mean temperature. As a non-linear function we assume a sigmoid function with
# 
# \begin{align}
# \alpha(T_i) = 0.3 \cdot (1-0.2 \cdot \tanh(0.5 \cdot (T_i-288))).
# \end{align}
# 
# Run the energy balance model for 100 years with four different initial conditions for T(0)=286.0, 287.9, 288.0, and 293.0 K, while fixing the other parameters to $C_w$= 2$\cdot10^8$ J/(m$^2 \cdot$ K) and $\tau_{mean}$=0.608. 
# 
# What can be said about the state of equilibrium?

# In[70]:


# Import some modules which are used in the function
import random         # Module to generate random number
import numpy as np    # Numpy


def ebm_ice_albedo(T0, Q=341.3, Cw=10e8, alpha=0.3, tau=0.64, years=100):
    ''' This is a simple Energy Balance Model including ice-albedo feedback.'''
  
    # Create result arrays (numpy) filled with zeros
    # Ts stores the temperature values, years the years since the beginning of
    # the simulation
    Ts    = np.zeros(years+1)
    Years = np.zeros(years+1)
    
    # Timestep in seconds (time step is 1 year)
    dt = 60*60*24*365                  # convert days to seconds

    # Initial and boundary conditions
    # Set the first value in the Ts to the initial condition
    Ts[0] = T0 

    # Integration over all years
    for n in range(years):
        # Parametrization of albedo. The albedo is a function of temperature.
        alpha_adapt = alpha * (1 - 0.2 * np.tanh(0.5*(Ts[n]-288)))
        # Store the number of iterations in the Years array
        Years[n+1] = n+1
        # Store the new temperature value in Ts
        Ts[n+1] = step_forward( Q, Ts[n], Cw, alpha_adapt, tau, dt )
        
    # Return both the temperature and year array
    return Years, Ts


# In[75]:


# Plot the albedo function
# Create an array containing the temperature range from 282 K to 295 K
T_range = np.arange(282,295,0.1)

# Plot the albedo function
plt.figure(figsize=(15,8))
plt.plot(T_range, 0.3 * (1 - 0.2 * np.tanh(0.5*(T_range-288))));


# In[76]:


# Run several ice-albedo feedback simulations with different initial conditions
yrs, Ts286 = ebm_ice_albedo(286.0, Q=342, Cw=2*10**8, alpha=0.30, tau=0.608, years=100)
yrs, Ts287 = ebm_ice_albedo(287.9, Q=342, Cw=2*10**8, alpha=0.30, tau=0.608, years=100)
yrs, Ts288 = ebm_ice_albedo(288.0, Q=342, Cw=2*10**8, alpha=0.30, tau=0.608, years=100)
yrs, Ts293 = ebm_ice_albedo(293.0, Q=342, Cw=2*10**8, alpha=0.30, tau=0.608, years=100)

# Plot the result
fig, ax = plt.subplots(1,1,figsize=(15,5))
ax.plot(yrs, Ts286); ax.plot(yrs, Ts287); ax.plot(yrs, Ts288); ax.plot(yrs, Ts293);


# **Extend the model with a simple ice/land use albedo parameterisation. (linear version)**
# 
# **Task 6:** In this parameterisation, the albedo is solely a function of mean temperature. We assume a simple linear relation according to
# 
# \begin{align}
#     f(x)= 
# \begin{cases}
#     \alpha_i,& \text{if } T\leq T_i\\
#     \alpha_g,& \text{if } T \geq T_g\\
#     \alpha_g+b(T_g-T) & \text{if } T_i<T<T_g
# \end{cases}
# \end{align}
# 
# with $T_i$=273 K, and $T_g$= 292 K. Run the energy balance model for 100 years with four different initial conditions for T(0)=286.0, 287.9, 288.0, and 293.0 K, while fixing the other parameters to $C_w$= 2$\cdot10^8$ J/(m$^2 \cdot$ K), and $\tau_{mean}$=0.608, $a_i$=0.6, and $a_g$=0.2. 
# 
# What can be said about the state of equilibrium?

# In[77]:


# Import some modules which are used in the function
import random         # Module to generate random number
import numpy as np    # Numpy


def ebm_ice_albedo_2(T0, Q=341.3, Cw=10e8, alpha=0.3, tau=0.608, years=100):
    ''' This is a simple Energy Balance Model with a linear ice-albedo 
    parametrization.'''
  
    # Create result arrays (numpy) filled with zeros
    # Ts stores the temperature values, years the years since the beginning of
    # the simulation
    Ts    = np.zeros(years+1)
    Years = np.zeros(years+1)
    
    # Timestep in seconds (time step is 1 year)
    dt = 60*60*24*365                  # convert days to seconds

    # Initial and boundary conditions
    Ts[0] = T0 # Set first value in the temperature array to the initial value
    a_i = 0.6  # This albedo value is used when the temperature < T_i
    a_g = 0.2  # This albedo value is used when the temperature > T_g
    T_i = 273  # This is the temperature threshold for snowball earth
    T_g = 292  # This is the temperature threshold when the earth is ice-free
    
    # Integrate over all years
    for n in range(years):
        # ice-albedo parametrization
        # if temperature is smaller equal T_i
        if Ts[n]<=T_i:
            # Set albedo to snowball earth albedo a_i
            alpha_adapt = a_i
        # if temperature is greater equal T_i
        elif Ts[n]>=T_g:
            # Set albedo to ice-free albedo a_g
            alpha_adapt = a_g
        # When temperature is between T_i and T_g
        else:
            # Calculate the parameter b 
            b = (a_i-a_g)/(T_g-T_i)
            # Calculate new albedo
            alpha_adapt = a_g + b*(T_g-Ts[n])

        # Store the number of iterations in the Years array
        Years[n+1] = n+1
        # Store the new temperature value in Ts
        Ts[n+1] = step_forward( Q, Ts[n], Cw, alpha_adapt, tau, dt )
        
    # return the Years and Ts arrays
    return Years, Ts


# In[79]:


# Run several ice-albedo simulations using different initial conditions
yrs, Ts286 = ebm_ice_albedo_2(280, Q=342, Cw=2*10**8, alpha=0.30, tau=0.608, years=100)
yrs, Ts287 = ebm_ice_albedo_2(287, Q=342, Cw=2*10**8, alpha=0.30, tau=0.608, years=100)
yrs, Ts288 = ebm_ice_albedo_2(288, Q=342, Cw=2*10**8, alpha=0.30, tau=0.608, years=100)
yrs, Ts293 = ebm_ice_albedo_2(293, Q=342, Cw=2*10**8, alpha=0.30, tau=0.608, years=100)

# Plot the results
fig, ax = plt.subplots(1,1,figsize=(15,8))
ax.plot(yrs, Ts286); ax.plot(yrs, Ts287); ax.plot(yrs, Ts288); ax.plot(yrs, Ts293);


# In[91]:


# Here we check the albedo parametrisation

def albedo(Ts):
    # Initial and boundary conditions
    a_i = 0.6  # This albedo value is used when the temperature < T_i
    a_g = 0.2  # This albedo value is used when the temperature > T_g
    T_i = 273  # This is the temperature threshold for snowball earth
    T_g = 292  # This is the temperature threshold when the earth is ice-free
    
    # ice-albedo parametrization
    # if temperature is smaller equal T_i
    if Ts<=T_i:
        # Set albedo to snowball earth albedo a_i
        alpha_adapt = a_i
    # if temperature is greater equal T_i
    elif Ts>=T_g:
        # Set albedo to ice-free albedo a_g
        alpha_adapt = a_g
    # When temperature is between T_i and T_g
    else:
        # Calculate the parameter b 
        b = (a_i-a_g)/(T_g-T_i)
        # Calculate new albedo
        alpha_adapt = a_g + b*(T_g-Ts)

    # Return albdo value
    return alpha_adapt

# Create an empty list which stores the albedo values
res = []

# Create an array containing the temperature range from 282 K to 295 K
T_range = np.arange(265,300,0.1)

# Fill the result list with albedos
for i in T_range:
    res.append(albedo(i))

# Plot the results
plt.plot(T_range,res);

