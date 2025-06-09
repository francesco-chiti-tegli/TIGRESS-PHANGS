import sys
sys.path.insert(0, '/export/home/extragal/lucia.armillotta/pyathena')
import pyathena as pa

from astropy import units
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
import h5py
from astropy.io import fits
from matplotlib.colors import LogNorm
from scipy.ndimage import shift
from pyathena.util.expand_domain import expand_xy
from scipy.integrate import quad
import matplotlib.transforms as transforms
from matplotlib.transforms import Bbox
from collections import Counter
import matplotlib.patches as patches

basedir = '/export/home/extragal/lucia.armillotta/TIGRESS/R8'
output_dir_Halpha = '/export/home/extragal/francesco.chititegli/TIGRESS_1/Halfa/'
output_dir_HI = '/export/home/extragal/francesco.chititegli/TIGRESS_1/H21maps/'
output_dir_IR = '/export/home/extragal/francesco.chititegli/TIGRESS_1/IRmaps/'
output_dir_exp = '/export/home/extragal/francesco.chititegli/TIGRESS_1/SPExpanded/'
output_dir_exp_history = '/export/home/extragal/francesco.chititegli/TIGRESS_1/History_SPExpanded/'
#tigress_dir = '/export/home/extragal/francesco.chititegli/TIGRESS_1/'
snapshot_number = 0

s = pa.LoadSim(basedir, verbose=False)
ds = s.load_vtk(num=s.nums[snapshot_number])
u = s.u

h = pa.read_hst(s.files['hst'])
print('Columns: ' + str(h.columns))

time_cf = (u.time).to(units.Gyr).value
mass_cf = (s.u.mass).to(units.M_sun).value
density_cf = (s.u.density).to(units.M_sun / units.pc**3).value
number_density_cf = (units.cm**-3).to(units.pc**-3)
solar_metallicity = 0.0134
dust_to_gas_ratio = 0.01
h_alpha_wave = 0.65628 # (micron)
h_alpha_energy = 3.03e-12 # (erg)
h_beta_wave = 0.48613 # (micron)
h_beta_energy = 4.09e-12 # (erg)
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
s.domain

# In Linzer+23 (1500 pc total), the total cell height is choosen as 375.
# In NORID (1024 pc total), the total cell height is choosen as 256.
# In Ridotto (512 pc total), the total cell height is choosen as 128.

tot_height_cells = 384
max_height = 768 + tot_height_cells // 2
min_height = 768 - tot_height_cells // 2

########################################################################################################################################################

timesteps = np.array(h['time'])
SFH10 = np.array(h['sfr10'])

def peaks_cav_finder(h, y_col='sfr10', time_col='time', prominence=1e-3):
    y = h[y_col].values
    x = h[time_col].values

    y_smooth = gaussian_filter1d(y, sigma=2)
    
    peaks_idx, _ = find_peaks(y_smooth, prominence=prominence)
    cav_idx, _ = find_peaks(-y_smooth, prominence=prominence)

    peaks = list(zip(x[peaks_idx], y[peaks_idx]))
    cav = list(zip(x[cav_idx], y[cav_idx]))

    return np.array(peaks), np.array(cav), np.array(y_smooth)

peaks_SFH, cavities_SFH, SFH10smooth = peaks_cav_finder(h)
print('Number of peaks in SFH: ' + str(len(peaks_SFH)))
print('Number of cavities in SFH: ' + str(len(cavities_SFH)))

beginning = 6
snapshot_indexes_to_export = list(range(beginning, beginning + 10))
snapshots_times_to_export = s.nums[beginning:beginning+10]

print('Times from ' + str(s.nums[beginning]) + ' Myr to ' + str(s.nums[beginning+9]) + ' Myr')
print('Indexes from ' + str(snapshot_indexes_to_export[0]) + ' to ' + str(snapshot_indexes_to_export[len(snapshot_indexes_to_export)-1]))

########################################################################################################################################################
########################################################################################################################################################

# Cell borders are always the same, regardless of the snapshot

xmin = ds.domain['le'][0] + dx[0] * np.arange(Nx[0])
ymin = ds.domain['le'][1] + dx[1] * np.arange(Nx[1])
zmin = ds.domain['le'][2] + dx[2] * np.arange(Nx[2])
xmax = xmin + dx[0]
ymax = ymin + dx[1]
zmax = zmin + dx[2]

# Expanding the cells borders
Nx_exp = np.array([3 * Nx[0], 3 * Nx[1], Nx[2]])
origin = ds.domain['le'] - Nx * dx
L_w = Nx * dx

xmin_exp = origin[0] + dx[0] * np.arange(Nx_exp[0])
ymin_exp = origin[1] + dx[1] * np.arange(Nx_exp[1])
zmin_exp = zmin
xmax_exp = xmin_exp + dx[0]
ymax_exp = ymin_exp + dx[1]
zmax_exp = zmax

# Borders of the simulation
print('X borders: ' + str(xmin_exp.min()) + ' pc  to  ' + str(xmax_exp.max()))
print('Y borders: ' + str(ymin_exp.min()) + ' pc  to  ' + str(ymax_exp.max()))
print('Z borders: ' + str(zmin_exp.min()) + ' pc  to  ' + str(zmax_exp.max()))
print('Number of cells: ' + str(Nx_exp))

########################################################################################################################################################

# In this code, I am still using the RIDOTTO slab, and SHEAR PERIODIC EXPANSION (SPExpansion)
# (-256 < x3 < 256 /// 704 < Nx[3] < 832) YES

# In the list snapshots_fields, every first entry corresponds to the snapshot number, starting from "beginning"
# Then, inside there is a list with the following entries: (0) electron density, (1) temperature, (2) gas_density, (3) HI_num_density, (4) proton density

# In the list snaphsots_stars, every first entry corresponds to the snapshot number, starting from "beginning"
# Then, inside there is an array with the stars in each snapshot

snapshots_fields = []
snaphsots_stars = []

for snapshot_number in snapshot_indexes_to_export:
    print('Snapshot number ' + str(snapshot_number) + ' at time ' + str(s.nums[snapshot_number]) + ' Myr')
    ds = s.load_vtk(num=s.nums[snapshot_number])

    # The fields
    electron_density = ds.get_field('ne')
    temperature = ds.get_field('T')
    gas_density = ds.get_field('rho')
    HI_num_density = ds.get_field('nHI')
    proton_density = ds.get_field('nHII')
    # And then SPExpand
    electron_density_exp = expand_xy(s,electron_density)
    temperature_exp = expand_xy(s,temperature)
    gas_density_exp = expand_xy(s,gas_density)
    HI_num_density_exp = expand_xy(s,HI_num_density)
    proton_density_exp = expand_xy(s,proton_density)
    print('Finished expanding fields in the snapshot number ' + str(snapshot_number))

    # The stars
    sp = s.load_starpar_vtk(num=s.nums_starpar[snapshot_number])
    positions_stars_unmasked = np.array(sp[['x1', 'x2', 'x3']])
    velocities_stars_unmasked = np.array(sp[['v1', 'v2', 'v3']])
    mass_stars_unmasked = np.array(sp['mass'])
    age_stars_unmasked = np.array(sp['age'])
    smoothing_length_stars_unmasked = np.full_like(mass_stars_unmasked, 0.5)
    metallicities_umasked = np.full_like(mass_stars_unmasked, solar_metallicity)
    stars_mask = (mass_stars_unmasked > 0) & (age_stars_unmasked > 0)
    positions_stars = positions_stars_unmasked[stars_mask]
    velocities_stars = velocities_stars_unmasked[stars_mask]
    mass_stars = mass_stars_unmasked[stars_mask]
    age_stars = age_stars_unmasked[stars_mask]
    smoothing_length_stars = smoothing_length_stars_unmasked[stars_mask]
    metallicities = metallicities_umasked[stars_mask]
    # And then SPExpand
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
            shifted_pos = positions_stars.copy()
            shifted_pos[:, 0] += (dx_shift)
            shifted_pos[:, 1] += (dy_shift + shear_shift_y)
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
    (positions_stars_exp_unmasked[:, 1] >= -1536) & (positions_stars_exp_unmasked[:, 1] <= 1536))
    positions_stars_exp = positions_stars_exp_unmasked[mask_inside_box]
    velocities_stars_exp = velocities_stars_exp_unmasked[mask_inside_box]
    mass_stars_exp = mass_stars_exp_unmasked[mask_inside_box]
    age_stars_exp = age_stars_exp_unmasked[mask_inside_box]
    smoothing_length_stars_exp = smoothing_length_stars_exp_unmasked[mask_inside_box]
    metallicities_exp = metallicities_exp_unmasked[mask_inside_box]
    print('Initial star number: ' + str(positions_stars.shape[0]) + '; expected after expansion: ' + str(positions_stars.shape[0] * 9))
    print('Pre-mask extended star number: ' + str(positions_stars_exp_unmasked.shape[0]))
    print('Masked extended star number: ' + str(positions_stars_exp.shape[0]))
    #Putting the stars in an array
    counter = 0
    for i in np.arange(positions_stars_exp.shape[0]):
        if (min_height * 4 - 3072 <= positions_stars_exp[i,2]):
            if (positions_stars_exp[i,2] < max_height * 4 - 3072):
                counter = counter + 1
    print(str(counter) + ' stars in the slab VS total of ' + str(positions_stars_exp.shape[0]))
    stars_array_exp = []
    for i in np.arange(positions_stars_exp.shape[0]):
        if (min_height * 4 - 3072 <= positions_stars_exp[i,2]):
            if (positions_stars_exp[i,2] < max_height * 4 - 3072):
                position_x = positions_stars_exp[i,0]
                position_y = positions_stars_exp[i,1]
                position_z = positions_stars_exp[i,2]
                smoothing_length = smoothing_length_stars_exp[i]
                mass = mass_stars_exp[i] * mass_cf * 1.5
                metal = metallicities_exp[i]
                age = age_stars_exp[i] * time_cf
                stars_array_exp.append([position_x, position_y, position_z, smoothing_length, mass, metal, age])
    stars_array_exp = np.array(stars_array_exp)
    
    # Putting everything inside the array for SFH snapshots
    snapshots_fields.append([electron_density_exp, temperature_exp, gas_density_exp, HI_num_density_exp, proton_density_exp])
    snaphsots_stars.append(stars_array_exp)
    print('Finished exporting snapshot number ' + str(snapshot_number))
    print()

########################################################################################################################################################
########################################################################################################################################################

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

########################################################################################################################################################

# Snapshot fields has [0] electron number density, [1] temperature, [2] gas density, [3] HI fraction

for indx, snapshot_number in enumerate(snapshot_indexes_to_export):
    print('NEW SNAPSHOT! Processing snapshot number ' + str(snapshot_number) + ', with relative index ' + str(indx))
    Halfa_array = []
    SED_array = []
    Hbeta_array = []
    old_disk_array = []
    dust_array = []
    hydrogen_array = []
    ID_number = 0
    
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
                
                #dust_density_val = (snapshots_fields[indx][2]['rho'].data)[k, j, i].astype(np.float64) * dust_to_gas_ratio
                #temperature_val = (snapshots_fields[indx][1]['T'].data)[k, j, i].astype(np.float64)
                #HI_ndens_val = (snapshots_fields[indx][3]['nHI'].data)[k, j, i].astype(np.float64)
                #metallicities_val = solar_metallicity * 0.03
                
                ne2 = (snapshots_fields[indx][0]['ne'].data)[k, j, i].astype(np.float64)
                T_4 = (snapshots_fields[indx][1]['T'].data)[k, j, i].astype(np.float64) / 1e4
                pe2 = (snapshots_fields[indx][4]['nHII'].data)[k, j, i].astype(np.float64)
                #h_alpha_luminosity_val = h_alpha_energy * (1.17e-13) * (T_4**(-0.942-0.030*np.log(T_4))) * (ne2 * pe2 * number_density_cf * volume)
                h_beta_luminosity_val = h_beta_energy * (3.03e-14) * (T_4**(-0.874-0.058*np.log(T_4))) * (ne2 * pe2 * number_density_cf * volume)
                
                #hydrogen_density_val = (HI_ndens_val) * H_mass_grams
                #H_surf_val = hydrogen_density_val * (units.g).to(units.M_sun) * number_density_cf * 4.0
                #HI_prefattore = (3/4) * 2.8843e-15 * 6.62e-27 * 3.0e8 / 21e-2
                #HI_lum_val = HI_prefattore * HI_ndens_val * number_density_cf * volume
                
                #star_disk_mass_val = mean_density_star_disk(xmin_val, ymin_val, zmin_val, xmax_val, ymax_val, zmax_val) * volume
                
                #Halfa_array.append([xmin_val, ymin_val, zmin_val, xmax_val, ymax_val, zmax_val, ID_number, dust_density_val])
                #SED_array.append([h_alpha_wave, h_alpha_luminosity_val, ID_number])
                Hbeta_array.append([h_beta_wave, h_beta_luminosity_val, ID_number])
                #dust_array.append([xmin_val, ymin_val, zmin_val, xmax_val, ymax_val, zmax_val, dust_density_val, temperature_val])          
                #hydrogen_array.append([xmin_val, ymin_val, zmin_val, xmax_val, ymax_val, zmax_val, hydrogen_density_val, metallicities_val, temperature_val, H_surf_val, HI_lum_val])
                #old_disk_array.append([xmin_val, ymin_val, zmin_val, xmax_val, ymax_val, zmax_val, star_disk_mass_val, solar_metallicity, 9.99])
                ID_number = ID_number + 1

    print("Final ID_Number: " + str(ID_number))

    #Halfa_array = np.array(Halfa_array)
    #SED_array = np.array(SED_array)
    Hbeta_array = np.array(Hbeta_array)
    #dust_array = np.array(dust_array)
    #hydrogen_array = np.array(hydrogen_array)
    #old_disk_array = np.array(old_disk_array)
    '''  
    print('Total mass analitically computed: ' + str(9 * 1024 * 1024 * 42 * 512 / np.sqrt(512**2 + 245**2)) + ' Msun')
    print('Total mass computed as mean density * volume * number cells: ' + str(np.mean(old_disk_array[:,6]) * (Nx[0] * Nx[1] * (max_height-min_height) * 9)) + ' Msun')
    print('Difference An - Num: ' + str((9 * 1024 * 1024 * 42 * 512 / np.sqrt(512**2 + 245**2)) - (np.mean(old_disk_array[:,6]) * (Nx[0] * Nx[1] * (max_height-min_height) * 9))) + ' Msun')
    print('Average metallicity: ' + str(np.mean(old_disk_array[:,7])))
    '''
    #######################################################################################################################

    true_output = output_dir_exp_history + 'Snap' + str(snapshot_number) + '/'

    #######################################################################################################################
    '''
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
    with open(f"{true_output}Halfa_Ridotto_exp.txt", "w") as txt_file:
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
    with open(f"{true_output}Dust_Ridotto_exp.txt", "w") as txt_file:
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
    with open(f"{true_output}Hydrogen_21_Ridotto_ConEmissione_exp.txt", "w") as txt_file:
        txt_file.write(header)
        np.savetxt(txt_file, hydrogen_array, fmt="%g")


    image_HI = np.sum((hydrogen_array[:, 10]).reshape(768, 768, 384), axis=-1).transpose()[256:512,256:512]
    surf_dens_HI = np.sum((hydrogen_array[:, 9]).reshape(768, 768, 384), axis=-1).transpose()[256:512,256:512]
    number_surf_dens_HI = surf_dens_HI * (units.M_sun).to(units.g) * (units.pc**-2).to(units.cm**-2) / H_mass_grams

    hdu = fits.PrimaryHDU(image_HI)
    hdu.header['BUNIT'] = 'erg/s'
    name_comment = 'HI Flux Density, Snap ' + str(snapshot_number)
    hdu.header['COMMENT'] = name_comment
    hdu.writeto(f"{true_output}HI_Flux_exp.fits", overwrite=True)

    hdu = fits.PrimaryHDU(number_surf_dens_HI)
    hdu.header['BUNIT'] = 'cm^-2'
    name_comment = 'Number surface density of HI, Snap ' + str(snapshot_number)
    hdu.header['COMMENT'] = name_comment
    hdu.writeto(f"{true_output}HI_Surf_exp.fits", overwrite=True)
    
    ########################################################################################################################
    
    header = """# Old_Stars_Ridotto_exp.txt: import file for cell source 
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

    with open(f"{true_output}Old_Stars_Ridotto_exp.txt", "w") as txt_file:
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
    with open(f"{true_output}Stars_Ridotto_exp.txt", "w") as txt_file:
        txt_file.write(header)
        np.savetxt(txt_file, snaphsots_stars[indx], fmt="%g")

    print('Finished saving arrays for snapshot number ' + str(snapshot_number) + ' in ' + str(true_output))

    ########################################################################################################################
    '''
    from pts.storedtable.io import writeStoredTable
    from pts.storedtable.io import readStoredTable
    def convertMonochromaticSimulation(wavelengths, luminosities, indexes, outFilePath):
        erg_to_watt = 1e-7
        luminosities_per_micron = luminosities * erg_to_watt
        w = np.array([0.656, 0.657])

        ID = indexes
        L = np.zeros((len(w), len(luminosities_per_micron)))
        L[0, :] = luminosities_per_micron / 0.656e-6
        L[1, :] = luminosities_per_micron / 0.657e-6

        writeStoredTable(outFilePath,['lambda', 'index'], ['m', '1'], ['lin', 'lin'],[w*1e-6, ID],['Llambda'], ['W/m'], ['lin'], [L])
    #convertMonochromaticSimulation(SED_array[:,0], SED_array[:,1], SED_array[:,2].astype(int), f"{true_output}SEDfamily_Ridotto_exp.stab")
    #readStoredTable(f"{true_output}SEDfamily_Ridotto_exp.stab")

    def convertMonochromaticSimulation_2(wavelengths, luminosities, indexes, outFilePath):
        erg_to_watt = 1e-7
        luminosities_per_micron = luminosities * erg_to_watt
        w = np.array([0.486, 0.487])

        ID = indexes
        L = np.zeros((len(w), len(luminosities_per_micron)))
        L[0, :] = luminosities_per_micron / 0.486e-6
        L[1, :] = luminosities_per_micron / 0.487e-6

        writeStoredTable(outFilePath,['lambda', 'index'], ['m', '1'], ['lin', 'lin'],[w*1e-6, ID],['Llambda'], ['W/m'], ['lin'], [L])
    convertMonochromaticSimulation_2(Hbeta_array[:,0], Hbeta_array[:,1], Hbeta_array[:,2].astype(int), f"{true_output}HbetaSED_Ridotto_exp.stab")
    readStoredTable(f"{true_output}HbetaSED_Ridotto_exp.stab")
    ########################################################################################################################

    print('Finished snapshot number ' + str(snapshot_number) + ', with relative index ' + str(indx))
    print()
