(nonlinearity_header)=
# Nonlinearity, Feedbacks and Predictability 

Many processes in nature are non-linear. These non-linearities can lead to
chaotic behaviour of systems that make deterministic prediction impossible. In
this exercise we will look at the non-linearities of processes and investigate
the behaviour of such systems. For simplicity, we will use the 'Von-May-Equation'.
This seemingly very simple equation helps us to analyse
apparently random solutions (chaos), which react sensitively to small perturbations.

### Learning objectives:
* A basic understanding of nonlinearity and chaos
* What are feedbacks
* Sensitivity of climate models
* Equilibrium climate sensitivity and feedback factors
* What is parametrization
* Equilibrium climate states
* What is hidden behind the term 'butterfly-effect'?

### After the exercise you should be able to answer the following questions:
* Why can we run climate simulations for many decades even though our predictability of weather events is very limited?
* Why are the initial conditions for fluid dynamic models so import?

### Python Notebooks
[Nonlinearity, Feedbacks and Predictability](nonlinearity:exercise)

[Solution: Nonlinearity, Feedbacks and Predictability](nonlinearity:solution)

### Problem description:
The starting point for our analysis is the 'Von-May-Equation', which is given by

$$
y_{t+1} = r \cdot y_{t} \cdot (1-y_{t}),
$$

with $r$ an pre-defined parameter and $y$ the function value at time $t$ and $t+1$. 

> **Task 1**: Write a function which solves the Von-May-Equation.

> **Task 2**: Run the code for several initial and parameter combination. What is particularly striking about increasing r-values?
> ```
> y(0)=0.5 and r=2.80 (alternatively, use y(0)=0.9) 
> y(0)=0.5 and r=3.30 (alternatively, use y(0)=0.9) 
> y(0)=0.5 and r=3.95 (alternatively, use y(0)=0.495) 
> y(0)=0.8 and r=2.80 
> ```

> **Task 3**: Extend this Von-May function by generating 20 random r-values and run
>   simulations with them. Sample the values from a normal distribution with
> mean 3.95 and standard deviation 0.015 (limit the r-values between 0 and 4). Then average over all time series. Plot
> both the time series, the averaged time series and the histogram of the
> averaged time series. What do you observe?



```{admonition} Revisit the EBM-Model
So far, the greenhouse effect has been parameterised by $\tau$ in the energy
balance model. However, the transmissivity (clouds etc.) fluctuates with the
 weather. At this point, the simple model does not account for this
dynamic. In order to include dynamics, we slightly modify the energy
balance model and generate a new $\tau$ at each time step. To do this, we
sample the transmission values from a normal distribution with a standard
deviation of 10 percent. 

> **Task 4**: Run the energy balance model T(0)=288  K, $C_w$ = 2$\cdot10^8$  J/(m$^2
> \cdot$ K), $\alpha$ =0.3, and $\tau_{mean}$=0.608 ($\pm$ 10%). Describe the time series.

> **Task 5**: Yet, the model does not take into account changes in albedo that result
>  from changes in glaciation and land use as a consequence of a changing
> climate. Therefore, we are now extending the model with a simple ice/land use
> albedo parameterisation. In this parameterisation, the albedo is solely a
> function of the mean temperature. To add some nonlinearity we assume a sigmoid function with 
>
>$$
\alpha(T_i) = 0.3 \cdot (1-0.2 \cdot \tanh(0.5 \cdot (T_i-288))).
$$
>
> (i) Plot the albedo function.
>
> (ii) Carry out the following simulations:
> - Run the energy balance model with four different initial conditions for
> T(0)=286.0, 287.9, 288.0, and 293.0 K, while fixing the other parameters to $C_w$ = 2$\cdot10^8$  J/(m$^2
>\cdot$ K) and $\tau_{mean}$ = 0.64%)
>
> What can be said about equilibrium climatic states?
>
> **Task 6**: Repeat the previous exercise with a linear parameterisation:
>
>$$
    f(x)= 
\begin{cases}
    \alpha_i,& \text{if } T\leq T_i\\
    \alpha_g,& \text{if } T \geq T_g\\
    \alpha_g+b(T_g-T) & \text{if } T_i<T<T_g
\end{cases}
$$ 
>
> **Task 7**: Determine the equilibrium climate sensitivity (ECS) and the feedback factor for the simulation from Task 5 for both with and without ice-albedo feedback. Is the feedback positive or negative? What can be said about the sign? 
>
> **Task 8**: Repeat the simulation from Task 5 (sigmoid function for albedo)
> with T(0)=289 K, but again sample the transmissivity on a normal distribution
> with a standard deviation of 10%.  What special feature can now be observed?
> What conclusions can be inferred regarding the prediction of weather and
> climate?
>
```



