"""
Calculate halo concentrations using universal model: c_vir = 6.39 * (D_zf * D_z * sigma_M)**1.26 + 2.79
"""
import numpy as np
import sys
import os

def universal_concentration(D_z, D_zf, sigma_M, a=6.39, b=1.26, c=2.79):
    """Calculate halo concentration c_vir"""
    return a * (D_zf * D_z * sigma_M)**b + c

def universal_D_zf(D_z, M, alpha=0.22, beta=0.11, gamma=0.35, beta1=-0.47):
    """Calculate formation growth factor D_zf"""
    return D_z * (alpha * (M/1e12)**beta + gamma * D_z**beta1)

def D_zf_eps_func(D_z, sigma_M, sigma_M_f, omega_f=0.75, delta_c=1.686):
    """Calculate D_zf using epsilon method"""
    Delta_S = np.sqrt(sigma_M_f**2 - sigma_M**2)
    return D_z / (1 + (omega_f / delta_c) * D_z * Delta_S)


def process_halo_data(input_file, z=0.0):
    """Process halo data and compute concentrations"""
    try:
        data = np.loadtxt(input_file, skiprows=1)
    except Exception as e:
        print(f"Error loading {input_file}: {e}")
        return None, None, None
    
    if len(data) == 0:
        print(f"Warning: No data in {input_file}")
        return None, None, None
    
    M = data[:, 0]
    z_data = data[:, 1]
    D_z = data[:, 2]
    sigma_M = data[:, 3]
    sigma_M_f = data[:, 4]
    sigma_M_f2 = data[:, 5]
    
    n_halos = len(M)
    results = np.zeros((n_halos, 9))
    
    for i in range(n_halos):
        D_zf_sim = universal_D_zf(D_z[i], M[i])
        D_zf_eps = D_zf_eps_func(D_z[i], sigma_M[i], sigma_M_f[i])
        D_zf_eps2 = D_zf_eps_func(D_z[i], sigma_M[i], sigma_M_f2[i])
        
        c_vir_sim = universal_concentration(D_z[i], D_zf_sim, sigma_M[i])
        c_vir_eps = universal_concentration(D_z[i], D_zf_eps, sigma_M[i])
        c_vir_eps2 = universal_concentration(D_z[i], D_zf_eps2, sigma_M[i])
        
        results[i] = [
            M[i], sigma_M[i], sigma_M_f[i], z_data[i], D_z[i],
            D_zf_sim, c_vir_sim, c_vir_eps, c_vir_eps2
        ]
    return M, D_z, results


def main():
    if len(sys.argv) < 4:
        sys.exit(1)
    
    input_file = sys.argv[1]
    cosmology = sys.argv[2]
    redshift = float(sys.argv[3])
    
    if not os.path.exists(input_file):
        print(f"Error: File '{input_file}' not found")
        sys.exit(1)
    
    output_file = f"halo_properties_{redshift}_{cosmology}.dat"
    
    M, D_z, results = process_halo_data(input_file, redshift)
    
    if results is None:
        print("Error: Failed to process data")
        sys.exit(1)
    
    # Print summary
    print("\n" + "="*50)
    print(f"Input:    {os.path.basename(input_file)}")
    print(f"Halos:    {len(M)}")
    print(f"M:     {M.min():.2e} - {M.max():.2e} M_sun/h")
    print(f"Cosmo:    {cosmology}")
    print(f"z:        {redshift}")
    print(f"Output:   {output_file}")
    print("="*50)
    
    # Save results
    header = (
        "# M[M_sun/h] sigma(M,0) sigma(0.5M,0) z D(z) "
        "D(z_f,peak) c_vir_sim c_vir_eps c_vir_eps2"
    )
    np.savetxt(output_file, results, fmt='%.6e %.6f %.6f %.3f %.6f %.6f %.6f %.6f %.6f',
               header=header, comments='')


if __name__ == '__main__':
    main()
