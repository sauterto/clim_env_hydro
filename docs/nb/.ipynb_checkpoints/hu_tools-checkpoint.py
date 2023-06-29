import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.optimize import minimize

import metpy.calc
from metpy.units import units

class HU_learning_material:

    def __init__(self):
        pass
    #--------------------------
    # Solar inclination angle
    #--------------------------
    def delta_s(self, d, dr, dy):
        """
        Solar inclination angle

        usage: delta_s(d, dr, dy)

        d  :: number of the day of the year
        dr :: day of the summer soltics (173 for non-leap years)
        dy :: number of days per year
        """

        return 0.409 * np.cos((2*np.pi*(d-dr))/(dy))


    #------------------------
    # Local elevation angle
    #------------------------
    def local_elevation_angle(self, lat, lon, UTC, d, dr, dy):
        """
        Local elevation angle

        usage: local_elevation_angle(lat, lon, UTC, d, dr, dy)

        lat :: latitude in degrees
        lon :: latitude in degrees
        UTC :: Coordinated universal time in hours
        d  :: number of the day of the year
        dr :: day of the summer soltics (173 for non-leap years)
        dy :: number of days per year
        """

        # Convert degrees in radiatn
        latr = np.deg2rad(lat)
        lonr = np.deg2rad(lon)

        # Get solar inclination angle
        phi_s = self.delta_s(d, dr, dy)
        return np.sin(latr)*np.sin(phi_s) - np.cos(latr)*np.cos(phi_s)*np.cos((np.pi*UTC/12)-lonr)


    #------------------
    # Transmissivity
    #------------------
    def tau(self, ch, cm, cl,lat, lon, UTC, d, dr, dy):
        """
        Parametrize the transmissivity of the atmosphere using

        usage: tau(ch, cm, cl,lat, lon, UTC, d, dr, dy)

        ch  :: cloud cover fractino of high clouds [-]
        cm  :: cloud cover fractino of middle clouds [-]
        cl  :: cloud cover fractino of low clouds [-]
        lat :: latitude in degrees
        lon :: latitude in degrees
        UTC :: Coordinated universal time in hours [h]
        d  :: number of the day of the year
        dr :: day of the summer soltics (173 for non-leap years)
        dy :: number of days per year
        """

        # Get the local elevation angle
        phi_s = self.local_elevation_angle(lat, lon, UTC, d, dr, dy)

        # Retrun the local elevation angle
        return (0.6+0.2*phi_s) * (1-0.4*ch) * (1-0.7*cm) * (1-0.4*cl)


    def SWout(self, SWin, albedo):
        """
        Calculates the outgoing shortwave radiation using the surface albedo

        usage: SWout(SWin, albedo)

        SWin   :: incoming shortwave radiation [W/mˆ2]
        albedo :: surface albedo [-]

        return: outgoing shortwave radiation
        """
        return (SWin * albedo)


    def LW(self, epsilon, T):
        """
        Radiated energy determined with the Stephan-Boltzmann law

        usage: LW(epsilon, T)

        epsilon :: black body emissivity [-]
        T       :: body temperature [K]
        """

        sigma = 5.67e-8
        return (epsilon * sigma * T**4)


    def lambda_max(self, T):
        """
        Wavelength of the emission maximum

        usage: lambda_max(T)

        T :: temperature [K]
        """
        return (2880/T)


    def EW(self, T):
        """
        Saturation vapor pressure

        usage: EW(T)

        Input:
            T   ::  Temperature [K]
        """
        if T >= 273.16:
            # over water
            Ew = 6.112 * np.exp((17.67*(T-273.16)) / ((T-29.66)))
        else:
            # over ice
            Ew = 6.112 * np.exp((22.46*(T-273.16)) / ((T-0.55)))
        return  Ew


    def emissivity(self, rH, Ta):
        """
        Atmospheric emissivity using the relative humidity and temperature

        usage: emissivity(rH, Ta)

        rH :: relative humdity [%]
        Ta :: Mean air temperature [K]
        """
        return 0.23 * 0.433 * ((rH*self.EW(Ta))/Ta)**(1/8)


    def heat_equation_loop(self,bc_surface, bc_bottom, depth, Nz, integration, dt, alpha):
        """"
        Solves the 1D-heat equation

        usage: heat_equation(,bc_surface, bc_bottom, depth, Nz, integration, dt, alpha)

        bc_surface :: boundary condition at the surface [K]
        bc_bottom  :: boundary condition at the bottom [K]
        depth      :: depth of the domain [m]
        Nz         :: number of grid points [-]
        integration:: number of iterations
        dt         :: time step [s]
        alpha      :: conductivity []

        """

        # Definitions
        dz    = depth/Nz  # Distance between grid points

        # Initialize temperature and depth field
        T = np.zeros(Nz)

        T[0] = bc_surface  # Set pen-ultima array to bc value (because the last grid cell
                              # is required to calculate the second order derivative)
        T[Nz-1] = bc_bottom      # Set the first elemnt to the bottom value

        # Create the solution vector for new timestep (we need to store the temperature values
        # at the old time step)
        Tnew = T.copy()

        t = 0
        # Loop over all times
        while t<=integration:

            # Loop over all grid points
            for z in range(1,Nz-1):
                Tnew[z] = T[z] + ((T[z+1] + T[z-1] - 2*T[z])/dz**2) \
                    * dt * alpha

            # Update old temperature array
            T = Tnew.copy()

            # Neumann boundary condition
            T[Nz-1] = T[Nz-2]

            # Update ellapsed time
            t = t+dt

        # return vertical temperature profile and grid spacing
        return T, dz


    def heat_equation(self, bc_surface, bc_bottom, depth, Nz, integration, dt, alpha):
        """
        Solves the heat equation using index arrays (faster than conventional solution)

        usage: heat_equation_indices(bc_surface, bc_bottom, depth, Nz, integration, dt, alpha)

        bc_surface :: boundary condition at the surface
        bc_bottom  :: boundary condition at the bottom
        depth      :: depth of the domain [m]
        Nz         :: number of grid points
        integration:: number of iterations
        dt         :: time step [s]
        """

        # Definitions
        dz    = depth/Nz # Distance between grid points

        # Define index arrays
        k = np.arange(1, Nz-1)
        kr = np.arange(2,Nz)
        kl = np.arange(0,Nz-2)


        # Initialize temperature and depth field
        T = np.zeros(Nz)

        T[0] = bc_surface     # Set pen-ultima array to bc value (because the last grid cell
                              # is required to calculate the second order derivative)
        T[Nz-1] = bc_bottom   # Set the first elemnt to the bottom value

        # Create the solution vector for new timestep (we need to store the temperature values
        # at the old time step)
        Tnew = T.copy()

        t = 0
        # Loop over all times
        while t<=integration:

            # ADD USER CODE HERE
            Tnew[k] = T[k] +((T[kr] + T[kl] - 2*T[k])/dz**2) *dt * alpha

            # Update old temperature array
            T = Tnew.copy()

            # Neumann boundary condition
            T[Nz-1] = T[Nz-2]

            # Update ellapsed time
            t = t+dt

        # return vertical temperature profile and grid spacing
        return T, dz

    def heat_equation_dt(self, T, dt, dz, alpha=1.2e-6):
        """
        This is an example of an time-dependent heat equation given a
        temperature signal at the surface and a time step dt. The heat equation is solved
        over the domain depth using Nz grid points.

        usage:  heat_equation_dt(ds, depth, dz, alpha=1.2e-6)

        T     :: Temperature profile [K]
        dz    :: Spacing between grid points [m]
        alpha :: soil themal diffusivity [mˆ2 sˆ-2]

        """

        # Define index arrays
        k  = np.arange(1,Nz-1)  # all indices at location i
        kr  = np.arange(2,Nz)   # all indices at location i+1
        kl  = np.arange(0,Nz-2) # all indices at location i-1

        # Create array for new temperature values
        Tnew = T

        # Set top BC - Dirlichet condition
        T[0] = value

        # Set lower BC - Neumann condition
        T[Nz-1] = T[Nz-2]

        # Update temperature using indices arrays
        Tnew[k] = T[k] + ((T[kr] + T[kl] - 2*T[k])/dz**2) * dt * alpha

        # return temperature array, grid spacing, and number of integration steps
        return Tnew

    def heat_equation_daily(self, ds, depth, dz, alpha=1.2e-6):
        """
        This is an example of an time-dependent heat equation given a
        temperature signal at the surface. The heat equation is solved
        over the domain depth using Nz grid points.

        usage:  heat_equation_daily(ds, depth, dz, alpha=1.2e-6)

        ds    :: Pandas temperature series [K]
        depth :: Domain depth [m]
        dz    :: Spacing between grid points [m]
        alpha :: soil themal diffusivity [mˆ2 sˆ-2]

        """

        # Get timestep in seconds
        dt = ds.index.freq.delta.total_seconds()

        # Definitions and assignments
        Nz  = int(depth/dz)        # Number of grid points
        dt  = 86400                # Time step in seconds (for each day)

        # Define index arrays
        k  = np.arange(1,Nz-1)  # all indices at location i
        kr  = np.arange(2,Nz)   # all indices at location i+1
        kl  = np.arange(0,Nz-2) # all indices at location i-1

        # Initial temperature field
        T = np.zeros(Nz)

        # Create array for new temperature values
        Tnew = T

        # 2D-Array containing the vertical profiles for all time steps (depth, time)
        T_all = np.zeros((Nz,len(ds.index)))

        # Loop over all times
        for index, value in enumerate(ds):

            # Set top BC - Dirlichet condition
            T[0] = value

            # Set lower BC - Neumann condition
            T[Nz-1] = T[Nz-2]

            # Update temperature using indices arrays
            Tnew[k] = T[k] + ((T[kr] + T[kl] - 2*T[k])/dz**2) * dt * alpha

            # Copy the new temperature als old timestep values (used for the
            # next time loop step)
            T = Tnew

            # Write result into the final array
            T_all[:,index] = Tnew


        # return temperature array, grid spacing, and number of integration steps
        return T_all, depth, dz

    # The following functions are needed for radiation method Moelg2009
    def solpars(self,lat):
        """ Calculate time correction due to orbital forcing (Becker 2001)
         and solar parameters that vary on daily basis (Mölg et al. 2003)
         0: day angle (rad); 1: in (deg); 2: eccentricity correction factor;
         3: solar declination (rad); 4: in (deg); 5: sunrise hour angle; 6: day length
        """

        timecorr = np.zeros((366, 4))
        solpars = np.zeros((366, 7))

        for j in np.arange(0, 365):
            # Time correction
            x = 0.9856 * (j + 1) - 2.72
            T2 = -7.66 * math.sin(math.radians(x)) - 9.87 * math.sin(
                2 * math.radians(x) + math.radians(24.99) + math.radians(3.83) * math.sin(math.radians(x)))
            timecorr[j, 0] = j + 1  # Julian Day
            timecorr[j, 1] = x
            timecorr[j, 2] = T2  # Time difference between True Local Time (TLT) and Average Local Time (ALT)
            timecorr[j, 3] = T2 * 15 / 60  # Time difference in deg (15°/h)

            # Solar parameters
            tau = 2 * math.pi * (j) / 365
            solpars[j, 0] = tau
            solpars[j, 1] = tau * 180 / math.pi
            solpars[j, 2] = 1.00011 + 0.034221 * math.cos(tau) + 0.00128 * math.sin(tau) + 0.000719 * math.cos(2*tau) + 0.000077 * math.sin(2 * tau)
            solpars[j, 3] = 0.006918 - 0.399912 * math.cos(tau) + 0.070257 * math.sin(tau) - 0.006758 * math.cos(2*tau) + 0.000907 * math.sin(2 * tau) - 0.002697 * math.cos(3 * tau) + 0.00148 * math.sin(3 * tau)
            solpars[j, 4] = solpars[j, 3] * 180 / math.pi
            solpars[j, 5] = math.acos(-math.tan(lat * math.pi / 180) * math.tan(solpars[j, 3])) * 180 / math.pi
            solpars[j, 6] = 2 / 15 * solpars[j, 5]

        # Duplicate line 365 for years with 366 days
        solpars[365, :] = solpars[364, :]
        timecorr[365, :] = timecorr[364, :]

        return solpars, timecorr


    def haversine(self, lat1, lon1, lat2, lon2):
        """ This function calculates the distance between two points given their longitudes and latitudes
         based on the haversine formula. """

        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        lon1_rad = math.radians(lon1)
        lon2_rad = math.radians(lon2)
        delta_lat = lat2_rad - lat1_rad
        delta_lon = lon2_rad - lon1_rad
        a = ((math.sin(delta_lat / 2)) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * (math.sin(delta_lon / 2)) ** 2) ** 0.5
        d = 2 * 6371000 * math.asin(a)
        return d


    def relshad(self, dem, mask, lats, lons, solh, sdirfn):
        """ This function calculates the topographic shading based on Mölg et al. 2009
         Input:
               dem:    A DEM of the study region, that also includes surrounding terrain
               mask:   A glacier mask
               lats:   The latitudes
               lons:   The longitudes
               solh:   The solar elevation (degrees)
               sdirfn: The illumination direction (in degrees from north)
         Output:
               illu:   Grid illu containing 0 = shaded, 1 = in sun
        """

        z = dem
        illu = dem * 0.0  # create grid that will be filled
        illu[:, :] = np.nan

        # Define maximum radius (of DEM area) in degreees lat/lon
        rmax = ((np.linalg.norm(np.max(lats) - np.min(lats))) ** 2 + (np.linalg.norm(np.max(lons) - np.min(lons))) ** 2) ** 0.5
        nums = abs(int(rmax * len(lats) / (lats[0] - lats[-1])))

        # Calculate direction to sun
        beta = math.radians(90 - sdirfn)
        dy = math.sin(beta) * rmax  # walk into sun direction (y) as far as rmax
        dx = math.cos(beta) * rmax  # walk into sun direction (x) as far as rmax

        # Extract profile to sun from each (glacier) grid point
        for ilat in np.arange(1, len(lats) - 1, 1):
            for ilon in np.arange(1, len(lons) - 1, 1):
                if mask[ilat, ilon] == 1:
                    start = (lats[ilat], lons[ilon])
                    targ = (start[0] + dy, start[1] + dx)  # find target position

                    # Points along profile (lat/lon)
                    lat_list = np.linspace(start[0], targ[0], nums)  # equally spread points along profile
                    lon_list = np.linspace(start[1], targ[1], nums)  # equally spread points along profile

                    # Don't walk outside DEM boundaries
                    lat_list_short = lat_list[(lat_list < max(lats)) & (lat_list > min(lats))]
                    lon_list_short = lon_list[(lon_list < max(lons)) & (lon_list > min(lons))]

                    # Cut to same extent
                    if (len(lat_list_short) > len(lon_list_short)):
                        lat_list_short = lat_list_short[0:len(lon_list_short)]
                    if (len(lon_list_short) > len(lat_list_short)):
                        lon_list_short = lon_list_short[0:len(lat_list_short)]

                    # Find indices (instead of lat/lon) at closets gridpoint
                    idy = (ilat, (np.abs(lats - lat_list_short[-1])).argmin())
                    idx = (ilon, (np.abs(lons - lon_list_short[-1])).argmin())

                    # Points along profile (indices)
                    y_list = np.round(np.linspace(idy[0], idy[1], len(lat_list_short)))
                    x_list = np.round(np.linspace(idx[0], idx[1], len(lon_list_short)))

                    # Calculate ALTITUDE along profile
                    zi = z[y_list.astype(int), x_list.astype(int)]

                    # Calclulate DISTANCE along profile
                    d_list = []
                    for j in range(len(lat_list_short)):
                        lat_p = lat_list_short[j]
                        lon_p = lon_list_short[j]
                        dp = haversine(start[0], start[1], lat_p, lon_p)
                        d_list.append(dp)
                    distance = np.array(d_list)

                    # Topography angle
                    Hang = np.degrees(np.arctan((zi[1:len(zi)] - zi[0]) / distance[1:len(distance)]))

                    if np.max(Hang) > solh:
                        illu[idy[0], idx[0]] = 0
                    else:
                        illu[idy[0], idx[0]] = 1

        return illu


    def LUTshad(self, solpars, timecorr, lat, elvgrid, maskgrid, lats, lons, STEP, TCART):
        """ This function calculates the look-up-table for topographic shading for one year.
         Input:
               solpars:   Solar parameters
               timecorr:  Time correction due to orbital forcing
               lat:       Latitude at AWS
               elvgrid:   DEM
               maksgrid:  Glacier mask
               lats:      Latitudes
               lons:      Longitudes
               STEP:      Time step (s)
               TCART:     Time correction due to difference MLT - TLT
         Output:
               shad1yr:   Look-up-table for topographic shading for 1 year
        """

        hour = np.arange(1, 25, 1)
        shad1yr = np.zeros((int(366 * (3600 / STEP) * 24), len(lats), len(lons)))  # Array (time,lat,lon)
        shad1yr[:, :, :] = np.nan

        # Go through days of year
        for doy in np.arange(0, 366, 1):

            soldec = solpars[doy, 3]  # solar declination (rad)
            eccorr = solpars[doy, 2]  # eccenctricy correction factor
            tcorr = timecorr[doy, 3]  # time correction factor (deg)

            # Go through hours of day
            for hod in np.arange(0, 24, int(STEP / 3600)):

                # calculate solar geometries
                stime = 180 + (15 / 2) - hod * 15 - tcorr + TCART
                sin_h = math.sin(soldec) * math.sin(lat * math.pi / 180) + math.cos(soldec) * math.cos(lat * math.pi / 180) * \
                        math.cos(stime * math.pi / 180)
                cos_sol_azi = (sin_h * math.sin(lat * math.pi / 180) - math.sin(soldec)) / math.cos(math.asin(sin_h)) / \
                        math.cos(lat * math.pi / 180)

                if stime > 0:
                    solar_az = math.acos(cos_sol_azi) * 180 / math.pi
                else:
                    solar_az = math.acos(cos_sol_azi) * 180 / math.pi * (-1)

                solar_h = math.asin(sin_h) * 180 / math.pi

                sdirfn = 180 - solar_az

                # Calculation (1 = in sun, 0 = shaded, -1 = night)
                if sin_h > 0.01:
                    illu = relshad(elvgrid, maskgrid, lats, lons, solar_h, sdirfn)
                    shad1yr[round(doy * (3600 / STEP) * 24 + (hod * 3600 / STEP)), maskgrid == 1] = illu[maskgrid == 1]
                else:
                    shad1yr[round(doy * (3600 / STEP) * 24 + (hod * 3600 / STEP)), maskgrid == 1] = -1.0

        return shad1yr


    def LUTsvf(self, lvgrid, maskgrid, slopegrid, aspectgrid, lats, lons):
        """ This function calculates the look-up-table for the sky-view-factor for one year.
         Input:
               elvgrid:   DEM
               maksgrid:  Glacier mask
               slopegrid: Slope
               aspectgrid:Aspect
               lats:      Latitudes
               lons:      Longitudes
        """

        slo = np.radians(slopegrid)
        asp = np.radians(aspectgrid)
        res = elvgrid * 0
        count = 0

        # Go through all directions (0-360°)
        for azi in np.arange(10, 370, 10):
            # Go through all elevations (0-90°)
            for el in np.arange(2, 90, 2):
                illu = relshad(elvgrid, maskgrid, lats, lons, el, azi)
                a = ((math.cos(np.radians(el)) * np.sin(slo) * np.cos(asp - np.radians(azi))) + (np.sin(np.radians(el)) * np.cos(slo)))
                a[a < 0] = 0
                a[a > 0] = 1
                a[illu == 0] = 0
                res = res + a
                count = count + 1

        vsky = elvgrid * 0
        vsky[:, :] = np.nan
        vsky[maskgrid == 1] = res[maskgrid == 1] / (36 * 44)

        return vsky


    def calcRad(self, solPars, timecorr, doy, hour, lat, tempgrid, pgrid, rhgrid, cldgrid, elvgrid, maskgrid, slopegrid,
                aspectgrid, shad1yr, gridsvf, STEP, TCART):
        """ This function computes the actual calculation of solar Radiation (direct + diffuse)
         including corrections for topographic shading and self-shading, based on Mölg et al. 2009, Iqbal 1983, Hastenrath 1984.
         Input:
               solpars:   Solar parameters
               timecorr:  Time correction due to orbital forcing
               doy:       Day of year
               hour:      Hour of day
               lat:       Latitude at AWSi
               tempgrid:  Air Temperature
               pgrid:     Air Pressure
               rhgrid:    Relative Humidity
               cldgrid:   Cloud fraction
               elvgrid:   DEM
               maksgrid:  Glacier mask
               slopegrid: Slope
               aspectgrid:Aspect
               shad1yr:   LUT topographic shading
               gridsvf:   LUT Sky-view-factor
               STEP:      Time step (s)
               TCART:     Time correction due to difference MLT - TLT
         Output:
               swiasky:   All-sky shortwave radiation
    """

        # Constants
        Sol0 = 1367          # Solar constant (W/m2)
        aesc1 = 0.87764      # Transmissivity due to aerosols at sea level
        aesc2 = 2.4845e-5    # Increase of aerosol transmissivity per meter altitude
        alphss = 0.9         # Aerosol single scattering albedo (Zhao & Li JGR112), unity (zero) -> all particle extinction is due to scattering (absorption)
        dirovc = 0.00        # Direct solar radiation at overcast conditions (as fraction of clear-sky dir. sol. rad, e.g. 10% = 0.1)
        dif1 = 4.6           # Diffuse radiation as percentage of potenial clear-sky GR at cld = 0
        difra = 0.66         # Diffuse radiation constant
        Cf = 0.65            # Constant that governs cloud impact

        soldec = solPars[doy - 1, 3]  # Solar declination (rad)
        eccorr = solPars[doy - 1, 2]  # Cccenctricy correction factor
        tcorr = timecorr[doy - 1, 3]  # Time correction factor (deg)

        # Output files
        swiasky = elvgrid.copy() + np.nan
        swidiff = elvgrid.copy() + np.nan

        # Mixing ratio from RH and Pres
        mixing_interp = metpy.calc.mixing_ratio_from_relative_humidity(pgrid * units.hPa, tempgrid * units.kelvin, rhgrid * units.percent)
        vp_interp = np.array(metpy.calc.vapor_pressure(pgrid * units.hPa, mixing_interp))

        # Solar geometries
        stime = 180 + (STEP / 3600 * 15 / 2) - hour * 15 - tcorr + TCART
        sin_h = math.sin(soldec) * math.sin(lat * math.pi / 180) + math.cos(soldec) * math.cos(
            lat * math.pi / 180) * math.cos(stime * math.pi / 180)
        mopt = 35 * (1224 * (sin_h) ** 2 + 1) ** (-0.5)
        if sin_h < 0:
            mopt = np.nan

        if sin_h > 0.01:  # Calculations are only performed when sun is there

            # Direct & diffuse radiation under clear-sky conditions
            TOAR = Sol0 * eccorr * sin_h
            TAUr = np.exp((-0.09030 * ((pgrid / 1013.25 * mopt) ** 0.84)) * (
                        1.0 + (pgrid / 1013.25 * mopt) - ((pgrid / 1013.25 * mopt) ** 1.01)))
            TAUg = np.exp(-0.0127 * mopt ** 0.26)
            k_aes = aesc2 * elvgrid + aesc1
            k_aes[k_aes > 1.0] = 1.0  # Aerosol factor: cannot be > 1
            TAUa = k_aes ** (mopt)
            TAUaa = 1.0 - (1.0 - alphss) * (1 - pgrid / 1013.25 * mopt + (pgrid / 1013.25 * mopt) ** (1.06)) * (1.0 - TAUa)
            TAUw = 1.0 - 2.4959 * mopt * (46.5 * vp_interp / tempgrid) / ((1.0 + 79.034 * mopt * (46.5 * vp_interp / tempgrid)) **
                                                                          0.6828 + 6.385 * mopt * (46.5 * vp_interp / tempgrid))
            taucs = TAUr * TAUg * TAUa * TAUw

            sdir = Sol0 * eccorr * sin_h * taucs  # Direct solar radiation on horizontal surface, clear-sky
            Dcs = difra * Sol0 * eccorr * sin_h * TAUg * TAUw * TAUaa * (1 - TAUr * TAUa / TAUaa) / (
                        1 - pgrid / 1013.25 * mopt + (pgrid / 1013.25 * mopt) ** (1.02))  # Diffuse solar radiation, clear sky
            grcs = sdir + Dcs  # Potential clear-sky global radiation

            # Correction for slope and aspect (Iqbal 1983)
            cos_zetap1 = (np.cos(np.radians(slopegrid)) * np.sin(np.radians(lat)) - np.cos(np.radians(lat)) * np.cos(np.radians(180 - aspectgrid)) *
                          np.sin(np.radians(slopegrid))) * np.sin(soldec)
            cos_zetap2 = (np.sin(np.radians(lat)) * np.cos(np.radians(180 - aspectgrid)) * np.sin(np.radians(slopegrid)) +
                          np.cos(np.radians(slopegrid)) * np.cos(math.radians(lat))) * np.cos(soldec) * np.cos(stime * np.pi / 180)
            cos_zetap3 = np.sin(np.radians(180 - aspectgrid)) * np.sin(np.radians(slopegrid)) * np.cos(soldec) * np.sin(stime * np.pi / 180)
            cos_zetap = cos_zetap1 + cos_zetap2 + cos_zetap3

            # Clear-sky direct solar radiation at surface (aspect & slope corrected)
            swidir0 = Sol0 * eccorr * cos_zetap * taucs
            swidir0[cos_zetap < 0.0] = 0.0  # self-shaded cells set to 0
            illu = elvgrid * 0.0
            illu = shad1yr[int(((doy - 1) * (86400 / STEP)) + (hour / (STEP / 3600))), :, :]
            swidir0[illu == 0.0] = 0.0
            sdir[illu == 0.0] = 0.0

            # Correction for cloud fraction
            swidiff[cldgrid > 0.0] = grcs[cldgrid > 0.0] * (((100 - Cf * 100) - dif1) / 100 * cldgrid[cldgrid > 0.0] +
                                        (dif1 / 100)) * gridsvf[cldgrid > 0.0]  # diffuse amount as percentage of direct rad.
            swidiff[cldgrid == 0.0] = Dcs[cldgrid == 0.0] * gridsvf[cldgrid == 0.0]
            swiasky[:, :] = swidir0 * (1 - (1 - dirovc) * cldgrid) + swidiff  # all-sky solar radiation at surface

        else:
            TOAR = 0.0
            swiasky[maskgrid == 1] = 0 * elvgrid[maskgrid == 1]
            illu = 0.0 * elvgrid - 1

        swiasky_ud = swiasky[::-1, :]

        return swiasky_ud

    def EB_fluxes(self, T0,T2,f0,f2,albedo,G,p,rho,U_L,z,z0):
        """ This function calculates the energy fluxes from the following quantities:

        Input:
        T_0       : Surface temperature, which is optimized [K]
        f         : Relative humdity as fraction, e.g. 0.7 [-]
        albedo    : Snow albedo [-]
        G         : Shortwave radiation [W m^-2]
        p         : Air pressure [hPa]
        rho       : Air denisty [kg m^-3]
        z         : Measurement height [m]
        z_0       : Roughness length [m]

        """

        # Some constants
        c_p = 1004.0      # specific heat [J kg^-1 K^-1]
        kappa = 0.40      # Von Karman constant
        sigma = 5.67e-8   # Stefan-Bolzmann constant

        # Aerodynamic roughness lengths
        z0t = z0/100  # sensible heat
        z0q = z0/10   # moisture

        # Bulk coefficients
        Cs_t = np.power(kappa,2.0) / ( np.log(z/z0t) * np.log(z/z0t) )
        Cs_q = np.power(kappa,2.0) / ( np.log(z/z0q) * np.log(z/z0q) )

        # Mixing ratio at measurement height and surface
        q2 = (f2 * 0.622 * (self.EW(T2) / (p - self.EW(T2))))
        q0 = (f0 * 0.622 * (self.EW(T0) / (p - self.EW(T0))))

        # Correction factor for incoming longwave radiation
        eps_cs = 0.23 + 0.433 * np.power(100*(f2*self.EW(T2))/T2,1.0/8.0)

        # Select the appropriate latent heat constant
        if T0<=273.16:
            L = 2.83e6 # latent heat for sublimation
        else:
            L = 2.50e6 # latent heat for vaporization

        # Calculate turbulent fluxes
        H_0 = rho * c_p * (1.0/0.8) * Cs_t * U_L * (T0-T2)
        E_0 = rho * L * (1.0/0.8) * Cs_q * U_L * (q0-q2)

        # Calculate radiation budget
        L_d = eps_cs * sigma * (T2)**4
        L_u = sigma * (T0)**4
        Q_0 = (1-albedo)*G + L_d - L_u

        return (Q_0,H_0,E_0,L_d,L_u,(1-albedo)*G)

    def optim_T0(self, x,T2,f0,f2,albedo,G,p,rho,U_L,z,z0,B):
        """ Optimization function for surface temperature:

        Input:
        T_0       : Surface temperature, which is optimized [K]
        f         : Relative humdity as fraction, e.g. 0.7 [-]
        albedo    : Snow albedo [-]
        G         : Shortwave radiation [W m^-2]
        p         : Air pressure [hPa]
        rho       : Air denisty [kg m^-3]
        z         : Measurement height [m]
        z_0       : Roughness length [m]

        """

        Q_0, H_0, E_0, L_d, L_u, SW_u = self.EB_fluxes(x,T2,f0,f2,albedo,G,p,rho,U_L,z,z0)

        # Get residual for optimization
        res = (Q_0-H_0-E_0+B)

        # guarantee that res-term is always positiv for optimization
        if res<0.0:
            res=9999

        # return the residuals
        return res

    def shf_bulk(self, rho,cp,Ch,U,Tz,T0):
        """ Sensible heat flux using bulk approach
            rho : Air density [kg m^-3]
            cp  : specific heat [J kg^-1 K^-1]
            Ch  : Bulk coefficient [-]
            U   : Wind velocity [m/s]
            Tz  : Temperature at height z
            T0  : Temperature at the surface

            Returns: Sensible heat flux
        """
        return rho*cp*Ch*U*(Tz-T0)

    def lhf_bulk(self, rho,L,Ce,U,qz,q0):
        """ Latent heat flux using bulk approach
            rho : Air density [kg m^-3]
            L   : Latent heat of vaporization [J kg^-1]
            Ch  : Bulk coefficient [-]
            U   : Wind velocity [m/s]
            qz  : Mixing ratio at height z [kg kg]
            q0  : Mixing ratio at the surface [kg kg]

            Returns: Latent heat flux
        """
        return rho*L*Ce*U*(qz-q0)

    def bulk_coeff_shf(self, z, z_0):
        """ Bulk coefficient for sensible heat
            z  : Measurement height [m]
            z0 : Roughness length [m]

            Returns: Bulk coefficient for sensible heat
        """
        kappa = 0.41
        # Aerodynamic roughness lengths
        z0t = z_0/100  # sensible heat
        return np.power(kappa,2.0) / ( np.log(z/z0t) * np.log(z/z0t) )

    def bulk_coeff_lhf(self, z, z_0):
        """ Bulk coefficient for latent heat
            z  : Measurement height [m]
            z0 : Roughness length [m]

            Returns: Bulk coefficient for latent heat
        """
        kappa = 0.41
        # Aerodynamic roughness lengths
        z0q = z_0/10   # moisture
        return np.power(kappa,2.0) / ( np.log(z/z0q) * np.log(z/z0q) )

    def mixing_ratio(self, f, T_a, p):
        """ Mixing ratio
        f  : Relative humidity [%]
        T_a: Air temperature [K]
        p  : Pressure [hPa]

        Returns: Mixing ratio
        """
        return (f * 0.622 * (self.EW(T_a) / (p - self.EW(T_a))))

    def get_surface_temp(self,T2,f0,f2,albedo,G,p,rho,U,z,z0,B):
        """ Optimize for surface temperature
        Input:
        T_0       : Surface temperature, which is optimized [K]
        f         : Relative humdity as fraction, e.g. 0.7 [-]
        albedo    : Snow albedo [-]
        G         : Shortwave radiation [W m^-2]
        p         : Air pressure [hPa]
        rho       : Air denisty [kg m^-3]
        z         : Measurement height [m]
        z_0       : Roughness length [m]
        B         : Ground heat flux [W m^-2]

        Returns: Surface temperature [K]
        """
        res = minimize(self.optim_T0,x0=285.0,args=(T2,f0,f2,albedo,G,p,rho,U,z,z0,B),bounds=((230,350.16),),\
           options={'maxiter':10000},method='L-BFGS-B',tol=1e-12)

        return res.x


class HU_moutain_waves:
    def __init__(self, var):
        """ Orographic Precipitation Model. The class requires a dictionary 'var' containing the variables, parameters and constants. The following values must be provided:
        DEM : 2D numpy array with the digital elevation mode
        Nx  : Number of cells in x-direction
        Ny  : Number of cells in y-direction
        dx  : Grid spacing in x-direction
        dy  : Grid spacing in y-direction
        ts  : Timestep in seconds
        lon : 1D-array with the longitude values
        lat : 1D-array with the latitude values
        output : String with the output filename
        U   : Zonal wind speed
        V.  : Meridional wind speed
        Cw. : Hydrometero conversion sensitivity factor
        N.  : Brunt-Väisälä frequency
        Hw. : Water vapor scale height
        tau : Advection time scale
        z.  : height above ground (for vertical velocity perturbation)
        
        returns:
        A netcdf file containing the topography, precipitation field, and vertical velocity fluctuations
        """
        self.DEM = var['DEM']
        self.Nx = var['Nx']
        self.Ny = var['Ny']
        self.dx = var['dx']
        self.dy = var['dy']
        self.ts = var['ts']
        self.lon = var['lon']
        self.lat = var['lat']
        self.ofile = var['output']
    
        self.U = var['U']
        self.V = var['V']
        self.Cw = var['Cw']
        self.N = var['N']
        self.Hw = var['Hw']
        self.tau = var['tau']
        self.zi = var['z']
    
        # Obtain wave number vectors
        self._calcFreq()


    def _calcFreq(self):
        """ Define the wave number vectors in x and y direction """
        # Define the wave vectors
        self.kx = np.empty(self.Nx)
        self.ky = np.empty(self.Ny)
        
        # Fill the wave vectors
        for i in range(0,self.Nx):
            _dfreq = float(i)/self.Nx
            _kx1, _intpart = np.modf((1.0/2.0) + _dfreq)
            _kx1 = _kx1 - (1.0/2.0)
            self.kx[i] = (_kx1*(2*np.pi/self.dx))
    
        for j in range(0,self.Ny):
            _dfreq = float(j)/self.Ny
            _ky1, _intpart = np.modf((1.0/2.0) + _dfreq)
            _ky1 = _ky1 - (1.0/2.0)
            self.ky[j] = (_ky1*(2*np.pi/self.dy))


    def opmRun(self):
        """ This is the actual OPM. 
        ts :: time step in seconds"""
    
        ds = xr.Dataset(
        data_vars=dict(
            HGT=(["lat", "lon"], self.DEM),
        ),
        coords=dict(
            lon=(["lon"], self.lon),
            lat=(["lat"], self.lat),
        ),
        attrs=dict(
            dx=self.dx,
            dy=self.dy)
        )
    
        self.ds =  ds
        _U = self.U
        _V = self.V
        _Cw = self.Cw
        _N = self.N
        _Hw = self.Hw
        _tau = self.tau
        
        # Define arrays for the fourier transformation
        cP = np.empty([self.Ny, self.Nx], dtype=complex)
        cW = np.empty([self.Ny, self.Nx], dtype=complex)

        # Fourier Transform of the DEM
        fft = np.fft.fft2(self.DEM)
 
        # Define the complex number
        z = complex(0, 1)
              
        #nan-times: precip = 0
        if np.isnan(_U*_V*_Cw*_N*_Hw*_tau):
            precip = np.zeros(np.shape(terrain))

        else:
            for i in range(0, self.Ny):
                for j in range(0, self.Nx):

                    # Lower boundary condition n*U=0
                    _wsurf = z * ((self.kx[j]*_U)+(self.ky[i]*_V)) * fft

                    # Intrisic frequency
                    _sigma = (self.kx[j]*_U) + (self.ky[i]*_V)

                    # Calculate vertical wave number
                    if (np.power(_N,2) < np.power(_sigma, 2)):
                        _m = z * np.sqrt( np.power(self.kx[j],2)+np.power(self.ky[i],2) )
                    elif (np.power(_N,2) > np.power(_sigma, 2) and np.power(_sigma, 2) > 1e-10):
                        _sign = 1 if (_U*self.kx[j] + _V*self.ky[i]) > 0 else -1 
                        _c =  (np.power(_N,2)/np.power(_sigma,2)) * (np.power(self.kx[j],2)+np.power(self.ky[i],2))
                        _m = np.sqrt(_c) * _sign
                    else:
                        _m = 0

                    # Check for unrealistic results
                    if (np.isnan(np.real(_m))):
                        _m = 0
                    if (np.isinf(np.real(_m))):
                        _m = 0

                    #if (kx[j] != 0 and ky[i] != 0 and np.power(sigma,2) !=0 and m != 0):
                    if (np.power(_sigma,2) !=0 and _m != 0):
                        cP[i,j] = (_Cw*_wsurf[i,j])/((1-(z*_m*_Hw)) * (1.0+(z*_sigma*_tau)) *
                                                     (1.0+(z*_sigma*_tau)))
                        cW[i,j] = (_wsurf[i,j]*np.exp(z*_m*self.zi))
                    else:
                        cW[i,j] = 0
                        cP[i,j]= 0

        self.ds['P'] = (('lat', 'lon'), np.maximum(np.copy(np.real(np.fft.ifft2(cP)))*self.ts, 0.0))
        self.ds['w'] = (('lat', 'lon'), np.copy(np.real(np.fft.ifft2(cW))))

        try:
            os.remove(self.ofile)
        except OSError:
            pass
        
        self.ds.to_netcdf(self.ofile,mode='w')
        xr.backends.file_manager.FILE_CACHE.clear()


