# Script to create a diagnostic diagram of all spaxels put together for AGN and
# inactives

import numpy as np
import astropy.io.fits as fits
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
import seaborn as sns

sns.set(style='ticks', color_codes=True)

# Directories
llama_dir = '/Users/ttshimiz/Dropbox/Research/LLAMA/'
proj_dir = llama_dir + 'projects/SINFONI_NIR_Emission/'
data_dir = proj_dir + 'ppxf_products/'
save_dir = proj_dir + 'line_ratio_maps/'

# Upload the sample properties
sample = pd.read_csv(llama_dir+'llama_general_table.csv', index_col=0,
                     header=0)
sample = sample.drop(['mcg-06-30-015_hband', 'mcg-06-30-015_kband',
                      'ngc4388', 'ngc2110', 'ngc1079'])

h2_bry_agn_sivi_detected = np.array([])
h2_bry_agn_nosivi = np.array([])
feii_bry_agn_sivi_detected = np.array([])
feii_bry_agn_nosivi = np.array([])
h2_bry_inactives = np.array([])
feii_bry_inactives = np.array([])

for name in sample.index:

        # Source name and files
        h2_file = data_dir + name + '/H2_1-0_S1_linefit/H2_1-0_S1_line_params_' + name + '_h2_1-0_s1_ppxf.fits'

        if sample.loc[name, 'broad_bry'] == 'No':
            bry_file = data_dir + name + '/Bry_linefit_narrow/Bry_line_params_' + name + '_bry_ppxf.fits'
        elif sample.loc[name, 'broad_bry'] == 'Yes':
            bry_file = data_dir + name + '/Bry_linefit_broad/Bry_narrow_line_params_' + name + '_bry_ppxf.fits'

        feii_file = data_dir + name + '/FeII_linefit/FeII_line_params_' + name + '_feii_ppxf.fits'

        # If its an AGN, upload the SiVI data
        if sample.loc[name, 'type'] == 'A':
            sivi_file = data_dir + name + '/SiVI_linefit/SiVI_line_params_' + name + '_sivi_ppxf.fits'
            sivi_hdu = fits.open(sivi_file)
            sivi_flux = sivi_hdu['int flux'].data
            sivi_flux_err = sivi_hdu['int flux error'].data
            sivi_vel = sivi_hdu['velocity'].data
            sivi_vel_err = sivi_hdu['velocity error'].data
            sivi_disp = sivi_hdu['velocity dispersion'].data
            sivi_disp_err = sivi_hdu['velocity dispersion error'].data

        # Import the data for H2, Bry, FeII
        h2_hdu = fits.open(h2_file)
        h2_flux = h2_hdu['int flux'].data
        h2_flux_err = h2_hdu['int flux error'].data
        h2_vel = h2_hdu['velocity'].data
        h2_vel_err = h2_hdu['velocity error'].data
        h2_disp = h2_hdu['velocity dispersion'].data
        h2_disp_err = h2_hdu['velocity dispersion error'].data

        bry_hdu = fits.open(bry_file)
        bry_flux = bry_hdu['int flux'].data
        bry_flux_err = bry_hdu['int flux error'].data
        bry_vel = bry_hdu['velocity'].data
        bry_vel_err = bry_hdu['velocity error'].data
        bry_disp = bry_hdu['velocity dispersion'].data
        bry_disp_err = bry_hdu['velocity dispersion error'].data

        feii_hdu = fits.open(feii_file)
        feii_flux = feii_hdu['int flux'].data
        feii_flux_err = feii_hdu['int flux error'].data
        feii_vel = feii_hdu['velocity'].data
        feii_vel_err = feii_hdu['velocity error'].data
        feii_disp = feii_hdu['velocity dispersion'].data
        feii_disp_err = feii_hdu['velocity dispersion error'].data

        # Create masks that remove bad fits and non-detections of the lines
        # We impose S/N > 3 for all 4 lines, velocity err < 50 km/s, and dispersion err < 50 km/s

        h2_mask = ((np.isfinite(h2_flux)) & (np.isfinite(h2_vel)) & (np.isfinite(h2_disp)) &
                   (h2_flux/h2_flux_err >= 3.0) & (h2_vel_err <= 50.0) & (h2_disp_err <= 50.0))
        h2_flux[~h2_mask] = np.nan

        bry_mask = ((np.isfinite(bry_flux)) & (np.isfinite(bry_vel)) & (np.isfinite(bry_disp)) &
                   (bry_flux/bry_flux_err >= 3.0) & (bry_vel_err <= 50.0) & (bry_disp_err <= 50.0))
        bry_flux[~bry_mask] = np.nan

        feii_mask = ((np.isfinite(feii_flux)) & (np.isfinite(feii_vel)) & (np.isfinite(feii_disp)) &
                   (feii_flux/feii_flux_err >= 3.0) & (feii_vel_err <= 50.0) & (feii_disp_err <= 50.0))
        feii_flux[~feii_mask] = np.nan

        if sample.loc[name, 'type'] == 'A':
            sivi_mask = ((np.isfinite(sivi_flux)) & (np.isfinite(sivi_vel)) & (np.isfinite(sivi_disp)) &
                        (sivi_flux/sivi_flux_err >= 3.0) & (sivi_vel_err <= 50.0) & (sivi_disp_err <= 50.0))
            sivi_flux[~sivi_mask] = np.nan

        # Calculate the H2/Bry and FeII/Bry line ratios
        mask_total = h2_mask & bry_mask & feii_mask
        line1 = h2_flux/bry_flux
        line1[~mask_total] = np.nan
        line2 = feii_flux/bry_flux
        line2[~mask_total] = np.nan

        # Add the SiVI detected pixels for each line ratio for AGN
        # And just combine all the pixels for each line ratio for inactives
        if sample.loc[name, 'type'] == 'A':

            h2_bry_agn_sivi_detected = np.hstack([h2_bry_agn_sivi_detected, line1[sivi_mask].flatten()])
            feii_bry_agn_sivi_detected = np.hstack([feii_bry_agn_sivi_detected, line2[sivi_mask].flatten()])
            h2_bry_agn_nosivi = np.hstack([h2_bry_agn_nosivi, line1[~sivi_mask].flatten()])
            feii_bry_agn_nosivi = np.hstack([feii_bry_agn_nosivi, line2[~sivi_mask].flatten()])

        elif sample.loc[name, 'type'] == 'I':

            h2_bry_inactives = np.hstack([h2_bry_inactives, line1.flatten()])
            feii_bry_inactives = np.hstack([feii_bry_inactives, line2.flatten()])


# Remove the Nans from the combined arrays
h2_bry_agn_sivi_detected = h2_bry_agn_sivi_detected[np.isfinite(h2_bry_agn_sivi_detected)]
feii_bry_agn_sivi_detected = feii_bry_agn_sivi_detected[np.isfinite(feii_bry_agn_sivi_detected)]
h2_bry_agn_nosivi = h2_bry_agn_nosivi[np.isfinite(h2_bry_agn_nosivi)]
feii_bry_agn_nosivi = feii_bry_agn_nosivi[np.isfinite(feii_bry_agn_nosivi)]
h2_bry_inactives = h2_bry_inactives[np.isfinite(h2_bry_inactives)]
feii_bry_inactives = feii_bry_inactives[np.isfinite(feii_bry_inactives)]


# Plot the diagnostic diagram
# SiVI detected spaxels
g = sns.jointplot(x=np.log10(h2_bry_agn_sivi_detected),
                  y=np.log10(feii_bry_agn_sivi_detected),
                  kind='kde', stat_func=None, color='r',
                  n_levels=100, shade=True, shade_lowest=False,
                  marginal_kws={"shade":False})

# Non-SiVI detected spaxels
sns.kdeplot(np.log10(h2_bry_agn_nosivi), data2=np.log10(feii_bry_agn_nosivi), shade=False,
            ax=g.ax_joint, shade_lowest=False, colors='g', cmap=None, linewidths=0.75)
sns.kdeplot(np.log10(h2_bry_agn_nosivi), color='g', ax=g.ax_marg_x)
sns.kdeplot(np.log10(feii_bry_agn_nosivi), color='g', ax=g.ax_marg_y, vertical=True)

# Inactive spaxels
g.ax_joint.scatter(np.log10(h2_bry_inactives), np.log10(feii_bry_inactives), s=30, color='b',
                   marker='+', linewidth=1.0)
sns.kdeplot(np.log10(h2_bry_inactives), color='b', ax=g.ax_marg_x)
sns.kdeplot(np.log10(feii_bry_inactives), color='b', ax=g.ax_marg_y, vertical=True)

# Plot the SB and AGN regions from Colina+2015
g.ax_joint.axvline(x=-0.1, ymax=1.3/2.7, color='tab:blue', ls='-', lw=1.2)
g.ax_joint.axhline(y=0.3, xmax=1.4/2.7, color='tab:blue', ls='-', lw=1.2)

g.ax_joint.axvline(x=-0.1, ymax=2.5/2.7, color='tab:red', ls='-', lw=1.2)
g.ax_joint.axvline(x=0.6, ymax=2.5/2.7, color='tab:red', ls='-', lw=1.2)
g.ax_joint.axhline(y=1.5, xmin=1.4/2.7, xmax=2.1/2.7, color='tab:red', ls='-', lw=1.2)

# Plot the line ratio for a "Fast J" shock model from Burton+1990
g.ax_joint.plot(np.log10(0.9), np.log10(4.35), marker='*', ms=20, mfc='y', mec='k')

g.ax_joint.set_xlim(-1.5, 1.2)
g.ax_joint.set_ylim(-1.0, 1.7)
g.ax_joint.set_xlabel(r'$\log$ H$_{2}$/Br$\gamma$')
g.ax_joint.set_ylabel(r'$\log$ [FeII]/Br$\gamma$')

g.fig.savefig(save_dir+'llama_diagnostic_diagram_combined.pdf', bbox_inches='tight')
