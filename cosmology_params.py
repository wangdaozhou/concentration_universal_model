"""
Cosmological Parameter Configurations
=====================================
This module provides pre-defined cosmological parameter sets for various
cosmological models, including ΛCDM, wCDM, SCDM, OCDM, scale-free, and WDM.
All parameters are defined at the present epoch (z=0) unless otherwise specified.
"""

cosmology_configs = {
    'LCDM': {
        'h': 0.6777,
        'omega_m': 0.307,
        'omega_b': 0.048,
        'n': 0.96,
        'sigma_8': 0.823,
        'delta_c': 1.686,
        'cosmology_type': 'CDM',
        'w_0': -1.0,
        'w_a': 0.0
    },
    'SCDM': {
        'h': 0.50,
        'omega_m': 1.0,
        'omega_b': 0.048,
        'n': 1.0,
        'sigma_8': 0.55,
        'delta_c': 1.686,
        'cosmology_type': 'CDM',
        'w_0': -1.0,
        'w_a': 0.0
    },
    'OCDM': {
        'h': 0.667,
        'omega_m': 0.30,
        'omega_b': 0.048,
        'n': 1.0,
        'sigma_8': 1.0,
        'delta_c': 1.686,
        'cosmology_type': 'OCDM',
        'w_0': -1.0,
        'w_a': 0.0
    },
    'SF_n2': {
        'h': 1.0,
        'omega_m': 1.0,
        'omega_b': 0.0,
        'n': -2.0,
        'sigma_8': 1.0,
        'delta_c': 1.686,
        'cosmology_type': 'EdS',
        'w_0': -1.0,
        'w_a': 0.0
    },
    'LWDM': {
        'h': 0.6766,
        'omega_m': 0.31,
        'omega_b': 0.048,
        'n': 1.0,
        'sigma_8': 1.0,
        'delta_c': 1.686,
        'cosmology_type': 'LWDM',
        'w_0': -1.0,
        'w_a': 0.0,
        'm_keV': 0.8
    },
    'wCDM': {
        'h': 0.5977,
        'omega_m': 0.4308,
        'omega_b': 0.048,
        'n': 0.9468,
        'sigma_8': 0.8161,
        'delta_c': 1.686,
        'cosmology_type': 'wCDM',
        'w_0': -0.816,
        'w_a': 0.0
    }
}