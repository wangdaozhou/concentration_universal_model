"""
Halo Mass Variance Calculator
==============================

This module provides a class for computing the variance of halo mass fluctuations
based on cosmological power spectra. It supports multiple cosmological models
including LCDM, SCDM, OCDM, LWDM, wCDM, and Scale-Free cosmologies.

Features:
---------
- Support for multiple cosmological models via configuration system
- Integration with CAMB for accurate power spectrum computation
- Interpolation tables for fast variance calculations
- Automatic adjustment of interpolation ranges based on input masses
- Vectorized operations for efficient batch processing
"""

import numpy as np
from scipy import integrate
import camb
from scipy.interpolate import interp1d
from GrowthFactor import linear_growth
from cosmology_params import cosmology_configs

class HaloMassVarianceCalculator:
    def __init__(self, cosmology_name='LCDM', table_size=1000, R_min=None, R_max=None, 
                 auto_adjust=True, safety_factor=0.1):
        """Initialize with a cosmology configuration name and interpolation table"""
        if cosmology_name not in cosmology_configs:
            raise ValueError(f"Unknown cosmology name: {cosmology_name}. Available options are: {list(cosmology_configs.keys())}")
        self.cosmo = cosmology_configs[cosmology_name]
        self.cosmo_name = cosmology_name
        # Lookup table parameters
        self.table_size = table_size
        self.auto_adjust = auto_adjust
        self.safety_factor = safety_factor
        # Initialize with default values, will be updated if auto_adjust is True
        self.R_min = R_min if R_min is not None else 0.001
        self.R_max = R_max if R_max is not None else 50.0
        self.R_table = np.logspace(np.log10(self.R_min), np.log10(self.R_max), table_size)   
        self.sigma2_table = None
        self.sigma_table_initialized = False
        # Store cosmology parameters for radius calculation
        self.rho_c = 3.0e4 / (8.0 * np.pi * 6.67430e-11) * 1.5513826e-2
        self.rho_0 = self.rho_c * self.cosmo['omega_m']
        # Precompute power spectrum using CAMB
        self._setup_power_spectrum()

    def _calculate_radius_from_mass(self, M):
        """Calculate radius from halo mass"""
        return (3.0 * M / (4.0 * np.pi * self.rho_0)) ** (1/3)

    def _adjust_table_bounds(self, masses):
        """Adjust R_min and R_max based on provided halo masses"""
        if not self.auto_adjust:
            return
        # Calculate radii for all masses
        radii = self._calculate_radius_from_mass(np.array(masses))
        # Find min and max radii with safety margin
        R_data_min = np.min(radii)
        R_data_max = np.max(radii)
        # Apply safety factor to extend bounds
        R_range = R_data_max - R_data_min
        self.R_min = max(0.001, R_data_min - self.safety_factor * R_range)
        self.R_max = R_data_max + self.safety_factor * R_range
        print(f"Auto-adjusted radius bounds: R_min={self.R_min:.4f}, R_max={self.R_max:.4f}")
        print(f"Data radius range: {R_data_min:.4f} - {R_data_max:.4f}")
        # Update table
        self.R_table = np.logspace(np.log10(self.R_min), np.log10(self.R_max), self.table_size)

    def _setup_power_spectrum(self):
        """Setup power spectrum using CAMB"""
        # For scale-free cosmologies, we use analytical form
        if self.cosmo_name in ['SF1_5', 'SF_n2']:
            self.power_spectrum_func = None
            return   
        # For other cosmologies, use CAMB
        params = self.cosmo
        # Set up CAMB parameters
        pars = camb.CAMBparams()
        if params['cosmology_type'] in ['CDM', 'OCDM', 'wCDM', 'LWDM']:
            pars.set_cosmology(
                H0=params['h'] * 100,
                ombh2=params['omega_b'] * params['h']**2,
                omch2=(params['omega_m'] - params['omega_b']) * params['h']**2,
                omk=0 if params['cosmology_type'] in ['CDM', 'LWDM', 'wCDM'] else 1 - params['omega_m'])
            # Set dark energy equation of state parameters
            if params['cosmology_type'] == 'wCDM':
                pars.set_dark_energy(w=params['w_0'], wa=params['w_a'])
            # Set initial power spectrum
            pars.InitPower.set_params(
                ns=params['n']
            )
        else:
            raise ValueError(f"Unsupported cosmology type: {params['cosmology_type']}")
        
        # Set power spectrum calculation parameters
        pars.set_matter_power(
            redshifts=[0.0],  
            kmax=100.0,       
            k_per_logint=0)
        # Calculate and get results
        results = camb.get_results(pars)
        # Get matter power spectrum
        kh, z, P_k = results.get_matter_power_spectrum(minkh=1e-4, maxkh=100.0, npoints=1000)
        P_k = P_k[0]  # Take z=0
        # For LWDM model, apply WDM correction to power spectrum
        if params['cosmology_type'] == 'LWDM':
            m_wdm_keV = params.get('m_keV', 0.8)
            Omega_wdm = params.get('omega_wdm', params['omega_m'])
            h = params['h']
            # Viel+2005 formula
            nu = 1.12
            alpha = 0.049 * (m_wdm_keV/1.0)**(-1.11) * (Omega_wdm/0.25)**(0.11) * (h/0.7)**(1.22)
            # WDM transfer function correction factor
            Tk = (1.0 + (alpha * kh)**(2.0 * nu))**(-5.0 / nu)
            P_k = P_k * Tk**2
        # Create interpolation function for the power spectrum
        self.power_spectrum_func = interp1d(kh, P_k, kind='cubic', 
                                           bounds_error=False, fill_value=(P_k[0], P_k[-1]))

    def _sigma2_integrand(self, k, R, AA):
        """Integrand function for sigma2 calculation"""
        kR = k * R
        window_f = 3.0 * ((np.sin(kR) - kR * np.cos(kR)) / (kR ** 3))
        if self.cosmo_name in ['SF_n2']:
            # Scale-free cosmology: P(k) = AA * k^n
            return (1.0 / (2.0 * np.pi * np.pi)) * AA * k**(2 + self.cosmo['n']) * window_f**2
        else:
            Pk_val = self.power_spectrum_func(k)
            return (1.0 / (2.0 * np.pi * np.pi)) * AA * k**2 * Pk_val * window_f**2

    def _setup_sigma2_table(self):
        """Precompute sigma2 values for lookup table"""
        print("Precomputing sigma2 lookup table...")
        print(f"Table range: R_min={self.R_min:.4f}, R_max={self.R_max:.4f}, size={self.table_size}")
        # First, normalize to sigma_8 with AA=1
        AA_temp = 1.0
        s2_8, _ = integrate.quad(self._sigma2_integrand, 0, np.inf, args=(8.0, AA_temp), 
                                epsabs=1.49e-6, epsrel=1.49e-6, limit=1000)
        self.AA_norm = self.cosmo['sigma_8'] ** 2 / s2_8
        # Precompute sigma2 for all R values in the table
        self.sigma2_table = np.zeros(self.table_size)
        for i, R in enumerate(self.R_table):
            result, _ = integrate.quad(self._sigma2_integrand, 0, np.inf, args=(R, AA_temp), 
                                     epsabs=1.49e-6, epsrel=1.49e-6, limit=1000)
            self.sigma2_table[i] = result * self.AA_norm
        # Create interpolation function
        self.sigma2_interp = interp1d(self.R_table, self.sigma2_table, kind='cubic', 
                                    bounds_error=False, fill_value=(self.sigma2_table[0], self.sigma2_table[-1]))
        self.sigma_table_initialized = True
        print("Sigma2 lookup table initialized successfully.")

    def initialize_for_masses(self, masses):
        """Initialize or reinitialize the lookup table for specific halo masses"""
        print("Initializing lookup table for provided halo masses...")
        # Adjust table bounds based on masses if auto_adjust is enabled
        if self.auto_adjust:
            self._adjust_table_bounds(masses)
        # Setup the sigma2 table
        self._setup_sigma2_table()

    def sigma2(self, R, AA=None):
        """Calculate sigma^2(R) using interpolation lookup table"""
        if not self.sigma_table_initialized:
            raise RuntimeError("Sigma2 lookup table not initialized. Call initialize_for_masses first.")
        # Ensure R is within table bounds and use interpolation
        R_clipped = np.clip(R, self.R_min, self.R_max)
        return self.sigma2_interp(R_clipped)

    def calculate_variance(self, Mvir, redshift, return_all=True):
        # Convert to array and ensure initialization
        Mvir = np.asarray(Mvir, dtype=np.float64)
        # Initialize lookup table if not already done
        if not self.sigma_table_initialized:
            self.initialize_for_masses(Mvir)
        # Get linear growth factor
        Dz, _ = linear_growth(
            redshift,
            h=self.cosmo['h'],
            omega_0=self.cosmo['omega_m'],
            w_0=self.cosmo['w_0'],
            w_a=self.cosmo['w_a'],
            cosmology=self.cosmo['cosmology_type']
        )
        # Initialize arrays for results
        valid_mask = ~np.isnan(Mvir)
        results = {
            'Mvir': Mvir,
            'D_z': np.full_like(Mvir, np.nan),
            'sigma_M': np.full_like(Mvir, np.nan),
            'sigma_M_f': np.full_like(Mvir, np.nan),
            'sigma_M_f2': np.full_like(Mvir, np.nan),
        }
        # Calculate for valid masses using vectorized operations
        if np.any(valid_mask):
            valid_Mvir = Mvir[valid_mask]
            # Compute radii (vectorized)
            R = self._calculate_radius_from_mass(valid_Mvir)
            Rf = self._calculate_radius_from_mass(0.5*valid_Mvir)
            Rf2 = self._calculate_radius_from_mass(0.14*valid_Mvir)
            # Calculate variances using lookup table (vectorized)
            sigma2_M = self.sigma2(R)
            sigma2_M_f = self.sigma2(Rf)
            sigma2_M_f2 = self.sigma2(Rf2)
            sigmaM = np.sqrt(sigma2_M)
            sigmaM_f = np.sqrt(sigma2_M_f)
            sigmaM_f2 = np.sqrt(sigma2_M_f2)
            # Store results
            results['D_z'][valid_mask] = Dz
            results['sigma_M'][valid_mask] = sigmaM
            results['sigma_M_f'][valid_mask] = sigmaM_f
            results['sigma_M_f2'][valid_mask] = sigmaM_f2
        return results if return_all else results['sigma_M']