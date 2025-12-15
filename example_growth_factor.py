
"""
An example for computing linear growth factors D(z) 
"""
import numpy as np
from GrowthFactor import linear_growth
from cosmology_params import cosmology_configs

def D_z(cosmo_name, z):
    cosmo = cosmology_configs[cosmo_name]
    
    # Handle array input: compute for each redshift element
    if isinstance(z, (list, tuple, np.ndarray)):
        return np.array([D_z(cosmo_name, zi) for zi in z])

    # Scalar input: single redshift calculation
    D, _ = linear_growth(
        z, 
        h=cosmo['h'],
        omega_0=cosmo['omega_m'],
        w_0=cosmo.get('w_0', -1.0),
        w_a=cosmo.get('w_a', 0.0),
        cosmology=cosmo.get('cosmology_type', 'CDM'))
    return D

# Example usage
if __name__ == "__main__":
    cosmology = 'LWDM'
    
    print('Cosmology:',cosmology)
    redshifts = np.linspace(0, 10, 11)
    D_array = D_z(cosmology, redshifts)
    
    print("Linear growth factors:")
    for z, D in zip(redshifts, D_array):
        print(f"  z={z:4.1f}: D(z)={D:.5f}")