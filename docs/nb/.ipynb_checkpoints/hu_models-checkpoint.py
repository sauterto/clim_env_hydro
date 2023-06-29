import os
import numpy as np
import pandas as pd
import math
import xarray as xr

class HU_mountain_waves:

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



