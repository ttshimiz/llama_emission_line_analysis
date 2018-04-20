import numpy as np
import astropy.io.fits as fits
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd

# Directories
llama_dir = '/Users/ttshimiz/Dropbox/Research/LLAMA/'
proj_dir = llama_dir + 'projects/SINFONI_NIR_Emission/'
data_dir = proj_dir + 'ppxf_products/'
save_dir = proj_dir + 'line_ratio_maps/all/'

# Upload the sample properties
sample = pd.read_csv(llama_dir+'llama_general_table.csv', index_col=0,
                     header=0)
sample = sample.drop(['mcg-06-30-015_hband', 'mcg-06-30-015_kband',
                      'ngc4388', 'ngc2110', 'ngc1079'])

for name in sample.index:

    if sample.loc[name, 'type'] == 'I':
		# Source name and files
		title_name = sample.loc[name, 'proper_name']
		h2_file = data_dir + name + '/H2_1-0_S1_linefit/H2_1-0_S1_line_params_' + name + '_h2_1-0_s1_ppxf.fits'

		if sample.loc[name, 'broad_bry'] == 'No':
			bry_file = data_dir + name + '/Bry_linefit_narrow/Bry_line_params_' + name + '_bry_ppxf.fits'
		elif sample.loc[name, 'broad_bry'] == 'Yes':
			bry_file = data_dir + name + '/Bry_linefit_broad/Bry_narrow_line_params_' + name + '_bry_ppxf.fits'

		feii_file = data_dir + name + '/FeII_linefit/FeII_line_params_' + name + '_feii_ppxf.fits'

		# Center pixel and pixel scale
		xcenter = sample.loc[name, 'center_x']
		ycenter = sample.loc[name, 'center_y']
		pixscale = np.float(sample.loc[name, 'pixscale'])/2e3     # arcsec/pixel

		# Import the data for H2, Bry, FeII, and SiVI line fits
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

		mask_total = h2_mask & bry_mask & feii_mask

		# Calculate the H2/Bry and FeII/Bry line ratios
		line1 = h2_flux/bry_flux
		line2 = feii_flux/bry_flux

		# Setup the figure
		# The first row will show the flux maps for the three lines
		# The second row will show the 2 line ratio maps and then the diagnostic diagram

		fig = plt.figure(figsize=(13, 8))
		fig.subplots_adjust(hspace=0.4, wspace=0.4)

		ax_h2 = fig.add_subplot(231)
		ax_feii = fig.add_subplot(232)
		ax_bry = fig.add_subplot(233)

		# Determine the offset in arcsecs from the central pixel
		nrows, ncols = h2_flux.shape
		extent = (-xcenter*pixscale, (ncols - xcenter)*pixscale, -ycenter*pixscale, (nrows - ycenter)*pixscale)

		# Plot the flux maps
		h2_im = ax_h2.imshow(h2_flux, origin='lower', cmap='viridis', extent=extent)
		feii_im = ax_feii.imshow(feii_flux, origin='lower', cmap='viridis', extent=extent)
		bry_im = ax_bry.imshow(bry_flux, origin='lower', cmap='viridis', extent=extent)

		ax_feii.tick_params(labelleft='off')
		ax_bry.tick_params(labelleft='off')

		ax_h2.set_title(title_name + r' H$_{2}$')
		ax_feii.set_title(title_name + ' [FeII]')
		ax_bry.set_title(title_name + r' Br$\gamma$')

		ax_h2.set_xlabel(r'$\Delta$X (arcsec)')
		ax_h2.set_ylabel(r'$\Delta$Y (arcsec)')
		ax_feii.set_xlabel(r'$\Delta$X (arcsec)')
		ax_bry.set_xlabel(r'$\Delta$Y (arcsec)')

		# Plot the line ratio maps
		ax_line1 = fig.add_subplot(234)
		ax_line2 = fig.add_subplot(235)

		line1_im = ax_line1.imshow(np.log10(line1), origin='lower', cmap='spectral_r', extent=extent)
		line2_im = ax_line2.imshow(np.log10(line2), origin='lower', cmap='spectral_r', extent=extent)

		ax_line2.tick_params(labelleft='off')
		divider1 = make_axes_locatable(ax_line1)
		divider2 = make_axes_locatable(ax_line2)
		cax1 = divider1.append_axes("right", size="5%", pad=0.05)
		cax2 = divider2.append_axes("right", size="5%", pad=0.05)
		line1_cb = plt.colorbar(line1_im, cax=cax1)
		line2_cb = plt.colorbar(line2_im, cax=cax2)

		ax_line1.set_title(r'$\log$ H$_{2}$/Br$\gamma$')
		ax_line2.set_title(r'$\log$ [FeII]/Br$\gamma$')
		ax_line1.set_xlabel(r'$\Delta$X (arcsec)')
		ax_line1.set_ylabel(r'$\Delta$Y (arcsec)')
		ax_line2.set_xlabel(r'$\Delta$X (arcsec)')


		# Plot the diagnostic diagram
		ax_diag = fig.add_subplot(236)
		ax_diag.yaxis.tick_right()
		ax_diag.yaxis.set_label_position('right')

		#ax_diag.scatter(np.log10(line1).flatten(), np.log10(line2).flatten(), s=5, marker='.', c='k')
		ax_diag.hexbin(np.log10(line1).flatten(), np.log10(line2).flatten(), gridsize=30, cmap='hot',
					   extent=(-1.5, 1.2, -1.0, 1.7), mincnt=5)

		ax_diag.set_xlim(-1.5, 1.2)
		ax_diag.set_ylim(-1.0, 1.7)
		ax_diag.set_xlabel(r'$\log$ H$_{2}$/Br$\gamma$')
		ax_diag.set_ylabel(r'$\log$ [FeII]/Br$\gamma$')
		ax_diag.set_title('Diagnostic')

		# Plot the SB and AGN regions from Colina+2015
		ax_diag.axvline(x=-0.1, ymax=1.3/2.7, color='tab:blue', ls='-', lw=1.0)
		ax_diag.axhline(y=0.3, xmax=1.4/2.7, color='tab:blue', ls='-', lw=1.0)

		ax_diag.axvline(x=-0.1, ymax=2.5/2.7, color='tab:red', ls='-', lw=1.0)
		ax_diag.axvline(x=0.6, ymax=2.5/2.7, color='tab:red', ls='-', lw=1.0)
		ax_diag.axhline(y=1.5, xmin=1.4/2.7, xmax=2.1/2.7, color='tab:red', ls='-', lw=1.0)
		ax_diag.text(-1.4, -0.9, 'SF', ha='left', va='bottom', color='tab:blue', fontsize=12)
		ax_diag.text(0.24, -0.9, 'AGN', ha='center', va='bottom', color='tab:red', fontsize=12)

		# Plot in a separate figure a map of discriminating pixels as either star-forming or AGN dominated
		fig2 = plt.figure(figsize=(8, 6))

		map = np.zeros(line1.shape)*np.nan

		ind_sf = (np.log10(line1) <= -0.1) & (np.log10(line2) <= 0.3)
		ind_agn = (np.log10(line2) <= 1.5) & (np.log10(line1) > -0.1) & (np.log10(line1) <= 0.6)
		ind_other = (~ind_sf) & (~ind_agn) & (mask_total)
		map[ind_sf] = 2.0
		map[ind_agn] = 1.0
		map[ind_other] = 3.0

		ax_map = fig2.add_subplot(111)
		map_im = ax_map.imshow(map, origin='lower', extent=extent, cmap='Set1', vmin=1.0, vmax=9.0)
		#map_cnt = ax_map.contour(sivi_flux, 5, origin='lower', extent=extent, colors='k')
		ax_map.set_xlabel(r'$\Delta$X (arcsec)')
		ax_map.set_ylabel(r'$\Delta$Y (arcsec)')
		ax_map.set_title(title_name)
		ax_map.text(0.05, 0.01, 'SF', ha='left', va='bottom', color='tab:blue', fontsize=20,
					transform=ax_map.transAxes)
		ax_map.text(0.15, 0.01, 'AGN', ha='left', va='bottom', color='tab:red', fontsize=20,
					transform=ax_map.transAxes)
		ax_map.text(0.3, 0.01, 'Other', ha='left', va='bottom', color='tab:green', fontsize=20,
					transform=ax_map.transAxes)

		fig.savefig(save_dir+name+'_line_ratio_maps_all.pdf', bbox_inches='tight')
		fig2.savefig(save_dir+name+'_diagnostic_regions_all.pdf', bbox_inches='tight')
		plt.close(fig)
		plt.close(fig2)
