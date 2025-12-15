"""
Halo Mass Variance Calculator with Adaptive Table Bounds
Usage: python calculate_halo_variance.py [mass_file.dat] [cosmology] [redshift]
"""
import numpy as np
import sys
import os
from VarianceCalculator import HaloMassVarianceCalculator
from cosmology_params import cosmology_configs

def load_masses(filename):
    """Load halo masses from input_masses.dat file"""
    masses = np.loadtxt(filename)
    return np.array([masses]) if masses.ndim == 0 else masses
    
def save_results(results, redshift, cosmology, filename="output_variance.dat"):
    """Save results to file with mixed formatting"""
    # 创建包含红移的列
    n_halos = len(results['Mvir'])
    redshifts_column = np.full(n_halos, redshift)
    
    output_data = np.column_stack([
        results['Mvir'], 
        redshifts_column,  # 添加红移列
        results['D_z'], 
        results['sigma_M'], 
        results['sigma_M_f'], 
        results['sigma_M_f2']
    ])
    
    np.savetxt(
        filename,
        output_data,
        fmt=['%.6e', '%.2f', '%.6f', '%.6f', '%.6f', '%.6f'],
        header=f'Mass[M_sun/h] z D(z) sigma(M,0) sigma(0.5M,0) sigma(0.14M,0)',
        comments='')
    
# Parse command line arguments
mass_file = sys.argv[1] if len(sys.argv) > 1 else "input_masses.dat"
cosmology = sys.argv[2] if len(sys.argv) > 2 else 'LCDM'
redshift = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
# Validate inputs
if cosmology not in cosmology_configs:
    print(f"Error: Cosmology '{cosmology}' not supported")
    sys.exit(1)
if not os.path.exists(mass_file):
    print(f"Error: Mass file {mass_file} not found")
    sys.exit(1)

# Load halo masses
masses = load_masses(mass_file)
print(f"Loaded {len(masses)} halos from {mass_file}")
print(f"Mass range: {np.min(masses):.3e} - {np.max(masses):.3e} M_sun/h")
print(f"Cosmology: {cosmology}, Redshift: z = {redshift}")

# Initialize calculator with auto-adjustment
calculator = HaloMassVarianceCalculator(
    cosmology_name=cosmology,
    auto_adjust=True,   
    safety_factor=0.1,   
    table_size=1000    
)

# Calculate mass variance (will auto-initialize table)
results = calculator.calculate_variance(masses, redshift)

# Preview results
print("\nPreview (first 5 halos):")
print("Mass [M_sun/h]   z    D(z)  sigma(M,0)  sigma(0.5M,0) sigma(0.14M,0)")
print("-" * 50)
for i in range(min(5, len(masses))):
    print(f"{results['Mvir'][i]:12.2e}  {redshift} {results['D_z'][i]:8.4f}  "
          f"{results['sigma_M'][i]:8.4f}  {results['sigma_M_f'][i]:8.4f} "
          f"{results['sigma_M_f2'][i]:8.4f}")
# Save full results
save_results(results, redshift, cosmology, filename="output_variance.dat")
print(f"\nFull results saved to output_variance.dat")