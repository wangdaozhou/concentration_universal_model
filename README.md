# Halo Mass Variance Calculator
# Calculate the linear growth factor
# Predicting halo formation and structure


##  Introduction

This code primarily serves two purposes:

1. To compute the linear growth factor ($D(z)$) , the root-mean-square (rms) fluctuation ($\sigma(M)$) of the linear density field for a halo of a given mass at z=0 under cosmological model.

2. To compute the linear growth factor $D(z_\mathrm{f,peak})$ at the halo formation epoch $z_\mathrm{f,peak}$, defined as the redshift when a halo reaches half of its final mass at redshift z, we use a universal fitting formula for $D(z_\mathrm{f,peak})$. This allows us to predict the concentration of dark matter halos of arbitrary mass, at any redshift, and for any cosmology. Our universal halo concentration model combines the linear growth factor with the rms fluctuation of the linear density field obtained in the first step. In addition, the halo formation epoch can be derived from either a semi-analytical model or an alternative fitting formula for the halo mass accretion history (MAH).

##  File Structure

-  `growth_factor.py` - Linear growth factor calculation by solving second-order differential equations (ODE).

-  `cosmology_params.py` - Cosmological parameter configurations, users can incorporate their own desired cosmology into this function and then calculate the variance of the corresponding simulated dark matter halos.
-  `VarianceCalculator.py` - Core mass variance calculation class.
-  `halo_variance.py` - Main program for variance calculation.
-  `formation_and_structure.py` - Calculates the halo formation redshift and concentration based on the dark halo mass accretion history and structure model.
-  `README.md` - Documentation
-  `input_masses.dat` - The required dataset of halo virial mass ($M_{vir}$).
-  `output_variance.dat` - The dataset of the rms fluctuation of the linear density field of halo.
-  `halo_properties_z_cosmology.dat` - The halo properties dataset includes: halo mass $M_{\mathrm{vir}}$, linear density fluctuations $\sigma(M)$ and $\sigma(M/2)$, redshift $z$, linear growth factors $D(z)$ and $D(z_{\mathrm{f},\mathrm{peak}})$, and halo concentrations $c_{\mathrm{vir}}$. Export the values ​​for three different concentrations: (1) $c_{\mathrm{vir},\mathrm{sim}}$ calculated using the universal fitting formula for $D(z_{\mathrm{f},\mathrm{peak}})$; (2) $c_{\mathrm{vir},\mathrm{eps}}$ and (3) $c_{\mathrm{vir},\mathrm{eps}2}$ derived from the analytical expression of $D(z_{\mathrm{f},\mathrm{peak}})$ with formation mass fractions $f = 0.5$ and $f = 0.14$, respectively.

**Note:​**​ Cosmology: LCDM, SCDM, OCDM, EdS, LWDM, wCDM.
  

##  Usage
First, you need to provide the halo mass `input_masses.dat` and the corresponding cosmological model `cosmology_params.py`. You can also add your desired cosmology in the `cosmology_params.py`. Then, use `growth_factor.py` and `halo_variance.py` to calculate $D(z)$ and $\sigma(M)$. Finally, use `formation_and_structure.py` to calculate the halo concentration parameters. The execution code is shown below.

###  Command Line Arguments
```bash
# Run calculation
python  halo_variance.py  input_masses.dat  [cosmology] [redshift]
python  formation_and_structure.py  output_variance.dat  [cosmology] [redshift]
# Examples
python  halo_variance.py  input_masses.dat  LCDM  0.0
python  formation_and_structure.py  output_variance.dat  LCDM  0.0
```

