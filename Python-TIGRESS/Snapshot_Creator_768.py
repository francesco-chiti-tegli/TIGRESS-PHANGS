# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 12:12:19 2025

@author: Francesco Chiti Tegli
"""

# This code (expansion non-Ridotto simulation box)
# And should be run in the tmux session called:          Norrbotten


# H alpha
# The SKIRT model for the H alpha is called:             HaTr3.ski
# And should be run in the tmux session called:          Dalarna


# 21 micron Dust
# The SKIRT model for the 21 micron Dust is called:      IRTr3.ski
# And should be run in the tmux session called:          Skane


# Importing the paths of the first snapshot of the simulation
# Importing the snapshot class and the units

import sys
sys.path.insert(0, '/export/home/extragal/lucia.armillotta/pyathena')
import pyathena as pa

from astropy import units
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
import h5py
from matplotlib.colors import LogNorm
from scipy.ndimage import shift
from pyathena.util.expand_domain import expand_xy
from scipy.integrate import quad
import matplotlib.transforms as transforms
from matplotlib.transforms import Bbox

basedir = '/export/home/extragal/lucia.armillotta/TIGRESS/R8'
output_dir_Halpha = '/export/home/extragal/francesco.chititegli/TIGRESS_1/Halfa/'
output_dir_HI = '/export/home/extragal/francesco.chititegli/TIGRESS_1/H21maps/'
output_dir_IR = '/export/home/extragal/francesco.chititegli/TIGRESS_1/IRmaps/'
output_dir_norid = '/export/home/extragal/francesco.chititegli/TIGRESS_1/SPExpanded_768/'

snapshot_number = 0

tot_height_cells = 384
max_height = 768 + tot_height_cells // 2
min_height = 768 - tot_height_cells // 2

print('Saving in ' + str(output_dir_norid))
print('Total height cells: ' + str(tot_height_cells))

s = pa.LoadSim(basedir, verbose=False)
ds = s.load_vtk(num=s.nums[snapshot_number])
u = s.u

# Lenghts are already in pc, Velocities are already in km/s, Temperatures are already be in K
# Gas cells' mass densities are already in g/cm3, and numeric densities are already in 1/cm3

time_cf = (u.time).to(units.Gyr).value
mass_cf = (s.u.mass).to(units.M_sun).value
density_cf = (s.u.density).to(units.M_sun / units.pc**3).value
number_density_cf = (units.cm**-3).to(units.pc**-3)
solar_metallicity = 0.0134
dust_to_gas_ratio = 0.01
h_alpha_wave = 0.65628 # (micron)
h_alpha_energy = 3.03e-12 # (erg)
H_mass = 8.41e-58 # (Msun)
H_mass_grams = 1.67e-24 # (grams)

print(s.basedir)
print(s.basename)
print(s.problem_id)
Nx = ds.domain['Nx']

# Printing the domain borders and cuboid cells grid

time_snapshot = ds.domain["time"]
Lxyz = ds.domain["Lx"]
qOmL = s.par["problem"]["qshear"]*s.par["problem"]["Omega"]*Lxyz[0]
qOmLt = qOmL * time_snapshot
dx = ds.domain['dx']
volume = dx[0] * dx[1] * dx[2]

# Cells borders (pc)
xmin = ds.domain['le'][0] + dx[0] * np.arange(Nx[0])
ymin = ds.domain['le'][1] + dx[1] * np.arange(Nx[1])
zmin = ds.domain['le'][2] + dx[2] * np.arange(Nx[2])
xmax = xmin + dx[0]
ymax = ymin + dx[1]
zmax = zmin + dx[2]

# Cells velocities (km/s)
velocities_x = ds.get_field('vx')
velocities_y = ds.get_field('vy')
velocities_z = ds.get_field('vz')

electron_density = ds.get_field('ne')  # Electrons Densities DERIVED FIELDS (1/cm3) -> (1/pc3) number_density_cf
temperature = ds.get_field('T')  # Temperatures (K)
gas_density = ds.get_field('rho')  # Gas density (g/cm3) -> (Msun/pc3) density_cf
HI_num_density = ds.get_field('nHI') # Neutral Hydrogen NUMBER density (1/cm3) -> (1/pc3) number_density_cf
H2_num_density = ds.get_field('nH2') # Molecular Hydrogen NUMBER density (1/cm3) -> (1/pc3) number_density_cf
hydrogen_fraction = ds.get_field('xHI') # Neutral hydrogen fraction

# Star particles
sp = s.load_starpar_vtk(num=s.nums_starpar[snapshot_number])

positions_stars_unmasked = np.array(sp[['x1', 'x2', 'x3']]) # Postions (pc)
velocities_stars_unmasked = np.array(sp[['v1', 'v2', 'v3']])  # Velocities (km/s)
mass_stars_unmasked = np.array(sp['mass'])  # Mass (Msun) -> (Msun) mass_cf
age_stars_unmasked = np.array(sp['age']) # Age (s) -> (Gyr) time_cf
smoothing_length_stars_unmasked = np.full_like(mass_stars_unmasked, 0.5) # Smoothing Lenght (4 pc) defaulted as the cells' edge
metallicities_umasked = np.full_like(mass_stars_unmasked, solar_metallicity) # Metallicity (Zsun) defaulted as the metallicity of the Sun

print('Min non-zero age: ' + str(np.min(age_stars_unmasked[age_stars_unmasked > 0])))
print('Min non-zero mass: ' + str(np.min(mass_stars_unmasked[mass_stars_unmasked > 0])))

# Some of them have negative age, or null mass, so we mask them out
stars_mask = (mass_stars_unmasked > 0) & (age_stars_unmasked > 0)
positions_stars = positions_stars_unmasked[stars_mask]
velocities_stars = velocities_stars_unmasked[stars_mask]
mass_stars = mass_stars_unmasked[stars_mask]
age_stars = age_stars_unmasked[stars_mask]
smoothing_length_stars = smoothing_length_stars_unmasked[stars_mask]
metallicities = metallicities_umasked[stars_mask]

print('Pre-mask shape: ' + str(positions_stars_unmasked.shape) + ' VS Post-mask shape: ' + str(positions_stars.shape))
print('Post-mask mass range [Msun]: ' + str(mass_stars.min() * mass_cf) + ' to ' + str(mass_stars.max() * mass_cf))
print('Post-mask age range [Gyr]: ' + str(age_stars.min() * time_cf) + ' to ' + str(age_stars.max() * time_cf))
print('Post-mask metal range [1]: ' + str(metallicities.min()) + ' to ' + str(metallicities.max()))

# RIDOTTO: we only take a thin slice of 512 pc
# For the first implementation of the full code, in order to test it as a whole, I only consider a slice of 256 pc
# centered on the galactic plane. With this we'll miss a big part of the less dense gas outside, but I will later modify it to import the full snapshot.
# (-512 < x3 < 512 /// 640 < Nx[3] < 896) YES

# Expanding the cells borders
Nx_exp = np.array([3 * Nx[0], 3 * Nx[1], Nx[2]])
origin = ds.domain['le'] - Nx * dx

xmin_exp = origin[0] + dx[0] * np.arange(Nx_exp[0])
ymin_exp = origin[1] + dx[1] * np.arange(Nx_exp[1])
zmin_exp = zmin
xmax_exp = xmin_exp + dx[0]
ymax_exp = ymin_exp + dx[1]
zmax_exp = zmax

print('X borders: ' + str(xmin_exp.min()) + ' pc  to  ' + str(xmax_exp.max()))
print('Y borders: ' + str(ymin_exp.min()) + ' pc  to  ' + str(ymax_exp.max()))
print('Z borders: ' + str(zmin_exp.min()) + ' pc  to  ' + str(zmax_exp.max()))
print('Number of cells: ' + str(Nx_exp))

# Expanding the extracted fields with the built-in function in Pyathena
electron_density_exp = expand_xy(s,electron_density)
temperature_exp = expand_xy(s,temperature)
gas_density_exp = expand_xy(s,gas_density)
HI_num_density_exp = expand_xy(s,HI_num_density)
H2_num_density_exp = expand_xy(s,H2_num_density)
hydrogen_fraction_exp = expand_xy(s,hydrogen_fraction)

# Expanding the star particles
L_w = Nx * dx

positions_stars_exp_unmasked = []
velocities_stars_exp_unmasked = []
mass_stars_exp_unmasked = []
age_stars_exp_unmasked_unmasked = []
smoothing_length_stars_exp_unmasked = []
metallicities_exp_unmasked = []

for dx_shift in [-3*L_w[0], -2*L_w[0], -L_w[0], 0., L_w[0], 2*L_w[0], 3*L_w[0]]:
    for dy_shift in [-3*L_w[0], -2*L_w[1], -L_w[1], 0., L_w[1], 2*L_w[1], 3*L_w[1]]:

        shifted_pos = positions_stars.copy()

        if (dx_shift==-L_w[0]):
            shear_shift_y = -qOmLt / dx[0]
        elif (dx_shift==L_w[0]):
            shear_shift_y = +qOmLt / dx[0]
        else:
            shear_shift_y = 0.

        total_dy = dy_shift

        shifted_pos = positions_stars.copy()
        shifted_pos[:, 0] += (dx_shift)
        shifted_pos[:, 1] += (total_dy + shear_shift_y)
        
        positions_stars_exp_unmasked.append(shifted_pos)
        velocities_stars_exp_unmasked.append(velocities_stars)
        mass_stars_exp_unmasked.append(mass_stars)
        age_stars_exp_unmasked_unmasked.append(age_stars)
        smoothing_length_stars_exp_unmasked.append(smoothing_length_stars)
        metallicities_exp_unmasked.append(metallicities)

positions_stars_exp_unmasked = np.array(np.vstack(positions_stars_exp_unmasked))
velocities_stars_exp_unmasked = np.array(np.vstack(velocities_stars_exp_unmasked))
mass_stars_exp_unmasked = np.array(np.hstack(mass_stars_exp_unmasked))
age_stars_exp_unmasked = np.array(np.hstack(age_stars_exp_unmasked_unmasked))
smoothing_length_stars_exp_unmasked = np.array(np.hstack(smoothing_length_stars_exp_unmasked))
metallicities_exp_unmasked = np.array(np.hstack(metallicities_exp_unmasked))

mask_inside_box = (
    (positions_stars_exp_unmasked[:, 0] >= -1536) & (positions_stars_exp_unmasked[:, 0] <= 1536) &
    (positions_stars_exp_unmasked[:, 1] >= -1536) & (positions_stars_exp_unmasked[:, 1] <= 1536)
)

positions_stars_exp = positions_stars_exp_unmasked[mask_inside_box]
velocities_stars_exp = velocities_stars_exp_unmasked[mask_inside_box]
mass_stars_exp = mass_stars_exp_unmasked[mask_inside_box]
age_stars_exp = age_stars_exp_unmasked[mask_inside_box]
smoothing_length_stars_exp = smoothing_length_stars_exp_unmasked[mask_inside_box]
metallicities_exp = metallicities_exp_unmasked[mask_inside_box]

print('Initial star number: ' + str(positions_stars.shape[0]) + '; expected after expansion: ' + str(positions_stars.shape[0] * 9))
print('Pre-mask extended star number: ' + str(positions_stars_exp_unmasked.shape[0]))
print('Masked extended star number: ' + str(positions_stars_exp.shape[0]))
print()
print('Initial number of cells: ' + str(Nx[0] * Nx[1] * (max_height-min_height)))
print('Expected number of cells: ' + str(Nx[0] * Nx[1] * (max_height-min_height) * 9))

# I computed the 21cm luminosity with a first-order approximation by employing the formula (8.3)
# by Bruce Draine (2011), page 71.
print('The pre-factor is: ' + str((3/4) * 2.8843e-15 * 6.62e-27 * 3.0e8 / 21e-2))

def mean_density_star_disk(xmin, ymin, zmin, xmax, ymax, zmax):
    sigma_star=42
    z_star=245

    def rho_z(z):
        return (sigma_star / 2) * (z_star**2) / ( (z**2 + z_star**2)**(3/2) )
        
    integral_rho, _ = quad(rho_z, zmin, zmax)
    mean_density = (integral_rho * dx[0] * dx[1]) / volume

    # the result is in Msun/pc**3
    return mean_density

# I arrange the data in several arrays: the spatial indexed distribution of the gas (Halfa_array) and its Halpha emission (SED_array); 
# the dust distribution (dust_array) for which I considered a solar neighbourhood D/G ratio; the hydrogen distribution (hydrogen_array)
# and its 21cm emission (HI_emission_array).
# The next step will be to import the velocities of the gas cells.

# Since SKIRT samples photons from SEDs, I had to index the cells and the luminosities.

Halfa_array = []
SED_array = []

old_disk_array = []

dust_array = []

hydrogen_array =  []
HI_emission_array = []

ID_number = 0

# Too change from normal to S-PBC expanded add:
# _exp
# Nx_exp instead of Nx

for i in range(Nx_exp[0]):
    print(i)
    for j in range(Nx_exp[1]):
        for k in range(min_height, max_height):
            xmin_val = xmin_exp[i]
            ymin_val = ymin_exp[j]
            zmin_val = zmin_exp[k]
            xmax_val = xmax_exp[i]
            ymax_val = ymax_exp[j]
            zmax_val = zmax_exp[k]
            
            dust_density_val = (gas_density_exp['rho'].data)[k, j, i].astype(np.float64) * dust_to_gas_ratio
            temperature_val = (temperature_exp['T'].data)[k, j, i].astype(np.float64)
            HI_ndens_val = (HI_num_density_exp['nHI'].data)[k, j, i].astype(np.float64)
            #H2_ndens_val = (H2_num_density_exp['nH2'].data)[k, j, i].astype(np.float64)
            metallicities_val = solar_metallicity

            ne2 = (electron_density_exp['ne'].data)[k, j, i].astype(np.float64)
            T_4 = (temperature_exp['T'].data)[k, j, i].astype(np.float64) / 1e4
            h_alpha_luminosity_val = h_alpha_energy * (1.17e-13 / (4*np.pi)) * (T_4**(-0.942-0.030*np.log(T_4))) * (ne2**2 * number_density_cf * volume)
            
            hydrogen_density_val = (HI_ndens_val) * H_mass_grams
            H_surf_val = hydrogen_density_val * (units.g).to(units.M_sun) * number_density_cf * 4.0

            HI_prefattore = (3/4) * 2.8843e-15 * 6.62e-27 * 3.0e8 / 21e-2
            HI_lum_val = HI_prefattore * HI_ndens_val * number_density_cf * volume

            star_disk_mass_val = mean_density_star_disk(xmin_val, ymin_val, zmin_val, xmax_val, ymax_val, zmax_val) * volume

            Halfa_array.append([xmin_val, ymin_val, zmin_val, xmax_val, ymax_val, zmax_val, ID_number, (dust_density_val / dust_to_gas_ratio)])
            SED_array.append([h_alpha_wave, h_alpha_luminosity_val, ID_number])
            dust_array.append([xmin_val, ymin_val, zmin_val, xmax_val, ymax_val, zmax_val, dust_density_val, temperature_val])          
            hydrogen_array.append([xmin_val, ymin_val, zmin_val, xmax_val, ymax_val, zmax_val, hydrogen_density_val, metallicities_val, temperature_val, H_surf_val])
            HI_emission_array.append([HI_lum_val])
            old_disk_array.append([xmin_val, ymin_val, zmin_val, xmax_val, ymax_val, zmax_val, star_disk_mass_val, solar_metallicity, 9.0])

            ID_number = ID_number + 1

print("Final ID_Number: " + str(ID_number))
del ID_number

# Putting everything in an array
Halfa_array = np.array(Halfa_array)
SED_array = np.array(SED_array)
dust_array = np.array(dust_array)
hydrogen_array = np.array(hydrogen_array)
HI_emission_array = np.array(HI_emission_array)
old_disk_array = np.array(old_disk_array)

# Checking old stars disk 
print('Total mass analitically computed: ' + str(9 * 1024 * 1024 * 42 * 1024 / np.sqrt(1024**2 + 245**2)) + ' Msun')
print('Total mass computed as mean density * volume * number cells: ' + str(np.mean(old_disk_array[:,6]) * (Nx[0] * Nx[1] * (max_height-min_height) * 9)) + ' Msun')
print('Difference An - Num: ' + str((9 * 1024 * 1024 * 42 * 1024 / np.sqrt(1024**2 + 245**2)) - (np.mean(old_disk_array[:,6]) * (Nx[0] * Nx[1] * (max_height-min_height) * 9))) + ' Msun')
print('Average metallicity: ' + str(np.mean(old_disk_array[:,7])))

counter = 0
for i in np.arange(positions_stars_exp.shape[0]):
        if (min_height * 4 - 3072 <= positions_stars_exp[i,2]):
            if (positions_stars_exp[i,2] < max_height * 4 - 3072):
                counter = counter + 1
print(str(counter) + ' stars in the area VS total of ' + str(positions_stars_exp.shape[0]))

stars_array_exp = []
for i in np.arange(positions_stars_exp.shape[0]):
    if (min_height * 4 - 3072 <= positions_stars_exp[i,2]):
        if (positions_stars_exp[i,2] < max_height * 4 - 3072):
            position_x = positions_stars_exp[i,0]
            position_y = positions_stars_exp[i,1]
            position_z = positions_stars_exp[i,2]
            smoothing_length = smoothing_length_stars_exp[i]
            
            #velocity_x = velocities_stars_exp[i,0]
            #velocity_y = velocities_stars_exp[i,1]
            #velocity_z = velocities_stars_exp[i,2]
            
            mass = mass_stars_exp[i] * mass_cf * 1.5 # SKIRT wants the initial mass, not the current mass, so I add +50%
            metal = metallicities_exp[i]
            age = age_stars_exp[i] * time_cf

            stars_array_exp.append([position_x, position_y, position_z, smoothing_length, mass, metal, age])
stars_array_exp = np.array(stars_array_exp)
print(stars_array_exp.shape)

# Now I save the arrays in .txt file (the next step will be to save them as .scol, which is the only other format
# that SKIRT can import sources and medium).

fmt_Halfa = ["%g", "%g", "%g", "%g", "%g", "%g", "%d", "%g"]
header = """# Halfa_Ridotto_exp.txt: import file for cell source -- Halfa
# Column 1: xmin (pc)
# Column 2: ymin (pc)
# Column 3: zmin (pc)
# Column 4: xmax (pc)
# Column 5: ymax (pc)
# Column 6: zmax (pc)
# Column 7: index (1)
# Column 8: mass density (g/cm3)
#
"""
with open(f"{output_dir_norid}Halfa_exp.txt", "w") as txt_file:
    txt_file.write(header)
    np.savetxt(txt_file, Halfa_array, fmt=fmt_Halfa)

#######################################################################################################################

header = """# Dust_Ridotto_exp.txt: import file for cell media -- dust
# Column 1: xmin (pc)
# Column 2: ymin (pc)
# Column 3: zmin (pc)
# Column 4: xmax (pc)
# Column 5: ymax (pc)
# Column 6: zmax (pc)
# Column 7: mass density (g/cm3)
# Column 8: temperature (K)
#
"""
with open(f"{output_dir_norid}Dust_exp.txt", "w") as txt_file:
    txt_file.write(header)
    np.savetxt(txt_file, dust_array, fmt="%g")

########################################################################################################################

header = """# Hydrogen_21_Ridotto_ConEmissione_exp.txt: import file for cell media -- gas
# Column 1: xmin (pc)
# Column 2: ymin (pc)
# Column 3: zmin (pc)
# Column 4: xmax (pc)
# Column 5: ymax (pc)
# Column 6: zmax (pc)
# Column 7: mass density (g/cm3)
# Column 8: metallicity (1)
# Column 9: temperature (K)
# Column 10: surface mass density (Msun/pc2)
# Column 11: HI luminosity (MJy/sr)
#
"""
with open(f"{output_dir_norid}Hydrogen_21_ConEmissione_exp.txt", "w") as txt_file:
    txt_file.write(header)
    np.savetxt(txt_file, np.hstack((hydrogen_array, HI_emission_array)), fmt="%g")

########################################################################################################################

header = """# Old_Stars_exp.txt: import file for cell source 
# Column 1: xmin (pc)
# Column 2: ymin (pc)
# Column 3: zmin (pc)
# Column 4: xmax (pc)
# Column 5: ymax (pc)
# Column 6: zmax (pc)
# Column 7: mass (Msun)
# Column 8: metallicity (1)
# Column 9: age (Gyr)
#
"""

with open(f"{output_dir_norid}Old_Stars_exp.txt", "w") as txt_file:
    txt_file.write(header)
    np.savetxt(txt_file, old_disk_array, fmt="%g")

########################################################################################################################

header = """# Stars_Ridotto_exp.txt: import file for particle source 
# Column 1: position x (pc)
# Column 2: position y (pc)
# Column 3: position z (pc)
# Column 4: smoothing length (pc)
# Column 5: mass (Msun)
# Column 6: metallicity (1)
# Column 7: age (Gyr)
#
"""
with open(f"{output_dir_norid}Stars_exp.txt", "w") as txt_file:
    txt_file.write(header)
    np.savetxt(txt_file, stars_array_exp, fmt="%g")

# The indexed SEDs are saved as .stab as it is the only format that SKIRT likes to import SEDs
# Furthermore, it is needed to convert from erg/s a watt

from pts.storedtable.io import writeStoredTable
def convertMonochromaticSimulation(wavelengths, luminosities, indexes, outFilePath):
    erg_to_watt = 1e-7
    luminosities_per_micron = luminosities * erg_to_watt
    w = np.array([0.656, 0.657])
    print(w)

    ID = indexes
    print(ID.shape)
    print(ID)
    print(np.arange(len(luminosities_per_micron)))
    
    L = np.zeros((len(w), len(luminosities_per_micron)))
    L[0, :] = luminosities_per_micron / 0.656e-6
    L[1, :] = luminosities_per_micron / 0.657e-6
    print(L.shape)
    print(L)

    writeStoredTable(outFilePath,['lambda', 'index'], ['m', '1'], ['lin', 'lin'],[w*1e-6, ID],['Llambda'], ['W/m'], ['lin'], [L])
    
convertMonochromaticSimulation(SED_array[:,0], SED_array[:,1], SED_array[:,2].astype(int), f"{output_dir_norid}SEDfamily_exp.stab")

from pts.storedtable.io import readStoredTable
readStoredTable(f"{output_dir_norid}SEDfamily_exp.stab")

print(output_dir_norid)


