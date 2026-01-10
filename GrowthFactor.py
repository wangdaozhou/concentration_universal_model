"""
Linear Growth Factor Calculator
===============================
This module provides functions for computing linear growth factors D(z) 
for various cosmological models by solving the linear growth differential equation.
"""

import numpy as np
from scipy.integrate import solve_ivp

def Hz(a, omega_0, h, w_0, w_a, cosmology):
    """
    Compute Hubble parameter for different cosmological models
    """
    H0 = h * 100.0 / 3.08567758e19  # H0 in 1/s
    if cosmology == 'EdS':
        return H0 * np.sqrt(1.0 / a**3)
    elif cosmology == 'OCDM':
        return H0 * np.sqrt(omega_0 / a**3 + (1.0 - omega_0) / a**2)
    else:
        return H0 * np.sqrt(omega_0 / a**3 + (1.0 - omega_0) * a**(-3 * (1 + w_0 + w_a)) * np.exp(-3 * w_a * (1 - a)))

def fcn(a, y, h, omega_0, w_0, w_a, cosmology):
    y1, y2 = y
    H0 = h * 100.0 / 3.08567758e19
    Hz_value = Hz(a, omega_0, h, w_0, w_a, cosmology)
    dy1_da = y2 / (a * Hz_value)
    dy2_da = -2.0 * (y2 / a) + 1.5 * omega_0 * H0 * H0 * y1 / (Hz_value * a**4)
    return [dy1_da, dy2_da]

def linear_growth(z, h, omega_0=0.3089, w_0=-1.0, w_a=0.0, cosmology='LCDM', tol=1e-6):
    """
    Compute linear growth factor D(z) for different cosmological models
    """
    if np.isscalar(z):
        z = np.array([float(z)])
    else:
        z = np.asarray(z, dtype=float)
    if cosmology == 'EdS':
        omega_0 = 1.0
    a_start = 1.0e-5
    y0 = [a_start, 0.0]
    fcn_wrapper = lambda a, y: fcn(a, y, h, omega_0, w_0, w_a, cosmology)
    
    sol_norm = solve_ivp(fcn_wrapper, [a_start, 1.0], y0, 
                         method='RK45', rtol=tol, atol=tol)
    norm = sol_norm.y[0, -1]
    
    a_values = 1.0 / (1.0 + z)
    D_z = np.zeros_like(z)
    ddot_z = np.zeros_like(z)
    
    sorted_indices = np.argsort(a_values)[::-1]
    sorted_a = a_values[sorted_indices]
    
    current_sol = solve_ivp(fcn_wrapper, [a_start, sorted_a[0]], y0, 
                           method='RK45', rtol=tol, atol=tol)
    
    for i, a_target in enumerate(sorted_a):
        if i > 0:
            current_sol = solve_ivp(fcn_wrapper, [sorted_a[i-1], a_target], 
                                   [current_sol.y[0, -1], current_sol.y[1, -1]], 
                                   method='RK45', rtol=tol, atol=tol)
        
        idx = sorted_indices[i]
        D_z[idx] = current_sol.y[0, -1] / norm
        ddot_z[idx] = current_sol.y[1, -1] / norm
    
    if len(D_z) == 1:
        return D_z[0], ddot_z[0]
    return D_z, ddot_z
