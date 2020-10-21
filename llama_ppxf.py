# Module for setting up and running PPXF for the LLAMA SINFONI cubes

from __future__ import (print_function, absolute_import)

# Standard library
import glob
from os import path
import pickle

# Third party libraries
import numpy as np
import astropy.io.fits as fits
import astropy.constants as c
import astropy.units as u
from ppxf import ppxf, ppxf_util
import matplotlib.pyplot as plt


# This library
#from .extern import ppxf, ppxf_util
from .lines import EMISSION_LINES
from .lines import ABSORPTION_LINES

# Common Constants
ckms = c.c.to(u.km/u.s).value


__all__ = ['create_emiles_templates', 'create_goodpixels', 'setup_fit', 'fit_single_spectrum',
           'fit_cube', 'construct_fit_products']


def create_emiles_templates(lam_range, fwhm_gal, velscale, ages, metals):
    """
    Uploads and modifies stellar template spectra from the EMILES library.
    Performs the following steps:
        1.) Cuts the spectra to the specified wavelengths
        2.) Log rebins the spectra to a specified velocity scale
        3.) Convolves the spectra to a certain instrumental resolution

    :param lam_range:
    :param ages:
    :param metals:
    :param fwhm_gal:
    :param velscale:
    :return:
    """

    # Use the EMILES SSP library as our template spectra
    file_dir = path.dirname(path.realpath(__file__))

    # Metallicities available in the EMILES library
    metals_avail = [-2.27, -1.79, -1.49, -1.26, -0.96, -0.66, -0.35, -0.25,
                    0.06, 0.15, 0.26, 0.4]

    # SSP ages available
    ages_avail = [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1,
                  0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
                  0.6, 0.7, 0.8, 0.9, 1.0, 1.25, 1.50, 1.75, 2.0,
                  2.25, 2.50, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.5,
                  5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5,
                  10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5, 14.0]

    # Gather the specified SSP spectra
    if (ages == 'all') & (metals == 'all'):

        emiles = glob.glob(file_dir + '/emiles_models/Pa00/EMILES_PADOVA00_BASE_CH_FITS/*.fits')

    elif (ages == 'all') & np.isscalar(metals):
        if metals in metals_avail:

            if metals < 0:
                mstr = 'm'+str(metals)
            else:
                mstr = 'p'+str(metals)

            emiles = glob.glob(file_dir + '/emiles_models/Pa00/EMILES_PADOVA00_BASE_CH_FITS/*'+mstr+'*.fits')

        else:
            raise ValueError('Specified metallicity not'
                             'part of EMILES library.')

    elif np.isscalar(ages) & (metals == 'all'):

        if ages in ages_avail:

            astr = '{0:07.4f}'.format(ages)
            emiles = glob.glob(
                file_dir + '/emiles_models/Pa00/EMILES_PADOVA00_BASE_CH_FITS/*T'+astr+'*.fits')

        else:
            raise ValueError('Specified SSP age not'
                             'part of EMILES library.')

    elif ages == 'all':

        emiles = []

        for m in metals:

            if m in metals_avail:
                if m < 0:
                    mstr = 'm' + str(metals)
                else:
                    mstr = 'p' + str(metals)

                emiles += glob.glob(file_dir + '/emiles_models/Pa00/EMILES_PADOVA00_BASE_CH_FITS/*'+mstr+'*.fits')

            else:
                raise ValueError('M/H = {0} not in EMILES library'.format(m))

    elif metals == 'all':

        emiles = []

        for a in ages:

            if a in ages_avail:
                astr = '{0:07.4f}'.format(a)
                emiles += glob.glob(file_dir + '/emiles_models/Pa00/EMILES_PADOVA00_BASE_CH_FITS/*T' + astr + '*.fits')
            else:
                raise ValueError(
                    'SSP Age = {0} not in EMILES library'.format(a))

    elif np.isscalar(ages) & np.isscalar(metals):

        if (ages in ages_avail) & (metals in metals_avail):

            if metals < 0:
                mstr = 'm' + str(metals)
            else:
                mstr = 'p' + str(metals)

            astr = '{0:07.4f}'.format(ages)

            emiles = glob.glob(
                file_dir + '/emiles_models/Pa00/EMILES_PADOVA00_BASE_CH_FITS/*Z'+mstr+'*T'+astr+'*.fits')

        else:
            raise ValueError('Check the SSP age and/or metallicity provided!')

    elif np.isscalar(ages):

        if ages in ages_avail:

            astr = '{0:07.4f}'.format(ages)
            emiles = []

            for m in metals:

                if m in metals_avail:
                    if m < 0:
                        mstr = 'm' + str(m)
                    else:
                        mstr = 'p' + str(m)

                    emiles += glob.glob(
                        file_dir + '/emiles_models/Pa00/EMILES_PADOVA00_BASE_CH_FITS/*Z' + mstr + '*T' + astr + '*.fits')

                else:
                    raise ValueError(
                        'M/H = {0} not in EMILES library'.format(m))
        else:
            raise ValueError('Specified SSP age not in EMILES library.')

    elif np.isscalar(metals):

        if metals in metals_avail:

            emiles = []

            if metals < 0:
                mstr = 'm' + str(metals)
            else:
                mstr = 'p' + str(metals)

            for a in ages:

                if a in ages_avail:
                    astr = '{0:07.4f}'.format(a)
                    emiles.append(
                        file_dir +
                        '/emiles_models/Pa00/EMILES_PADOVA00_BASE_CH_FITS/Ech1.30Z' +
                        mstr + 'T' + astr +
                        '_iTp0.00_baseFe.fits')

                else:
                    raise ValueError(
                        'SSP Age = {0} not in EMILES library'.format(a))
        else:
            raise ValueError(
                'M/H = {0} not in EMILES library'.format(metals))

    else:

        emiles = []

        for a in ages:

            if a in ages_avail:
                astr = '{0:07.4f}'.format(a)
            else:
                raise ValueError(
                    'SSP Age = {0} not in EMILES library'.format(a))

            for m in metals:

                if m in metals_avail:
                    if m < 0:
                        mstr = 'm' + str(m)
                    else:
                        mstr = 'p' + str(m)
                else:
                    raise ValueError(
                        'M/H = {0} not in EMILES library'.format(m))

                emiles.append(file_dir + '/emiles_models/Pa00/EMILES_PADOVA00_BASE_CH_FITS/Ech1.30Z' + mstr + 'T' + astr +
                              '_iTp0.00_baseFe.fits')

    print('Using {0} stellar templates.'.format(len(emiles)))

    # Extract the wavelength range and logarithmically rebin one spectrum
    # to a velocity scale equal to the galaxy spectrum, to determine
    # the size needed for the array which will contain the template spectra.
    spec0 = fits.getdata(emiles[0])
    header = fits.getheader(emiles[0])
    lam_emiles = (np.arange(header['NAXIS1'])*header['CDELT1'] +
                  header['CRVAL1'])
    lam_emiles = lam_emiles/1e4      # Convert from Angstrom to micron

    mask = (lam_emiles >= lam_range[0]) & (lam_emiles <= lam_range[1])
    spec0 = spec0[mask]
    lam_emiles = lam_emiles[mask]
    spec0_new, log_lam, velscale = ppxf_util.log_rebin([lam_emiles[0], lam_emiles[-1]],
                                                       spec0, velscale=velscale)
    templates = np.empty((spec0_new.size, len(emiles)))

    # Convolve each stellar template with a Gaussian to match the instrumental resolution
    # of our data. Then log rebin each one and store in "templates".
    fwhm_emiles = 2.35482*60.
    sigma_kms = np.sqrt(fwhm_gal ** 2 - fwhm_emiles ** 2) / 2.355
    sigma_pix = sigma_kms/ckms * lam_emiles / (lam_emiles[1] - lam_emiles[0])

    for i, f in enumerate(emiles):

        hdu = fits.open(f)
        spec = hdu[0].data
        spec_cut = spec[mask]
        spec_conv = ppxf_util.gaussian_filter1d(spec_cut, sigma_pix)
        spec_new, log_lam, velscale_temp = ppxf_util.log_rebin([lam_emiles[0], lam_emiles[-1]],
                                                               spec_conv, velscale=velscale)
        templates[:, i] = spec_new / np.median(spec_new)

    # Remove any templates that are just NaNs
    nan_temps = np.any(np.isnan(templates), axis=0)
    temps = templates[:, ~nan_temps]

    print('Removed {0} templates.'.format(np.sum(nan_temps)))
    print('Now using {0} templates.'.format(temps.shape[1]))

    return temps, log_lam


def create_goodpixels(loglam, lam_range_temp, z=0, dv=800.,
                      mask_h2_lines=True, mask_ionized_lines=True,
                      mask_broad_bry=False, mask_stellar_bry=False,
                      smooth=False, mask_telluric=False, mask_hband_bry=False):
    """
    Create the list of "good" pixels to use in the PPXF fit. Automatically masks out several pixels
    in the beginning and the end as well as the region between H and K band with low transmission.
    :param loglam: Log of the wavelengths for the spectra to be fit
    :param lam_range_temp: Two element vector with the beginning and end of the stellar template
        range.
    :param z: Estimated redshift of the galaxy
    :param dv: Halfwidth around center of emission lines to mask out in km/s
    :param mask_h2_lines: True or False as to whether to mask out prominent H2 lines
    :param mask_ionized_lines: True or False as to whether to mask out prominent ionized lines
    :param mask_broad_bry: True or False as to whether Bry is expected to be broad
    :param mask_stellar_bry: True or False as to whether to mask residual stellar Bry lines
    :return: goodpixels: List of indices to use in the fit
    """

    # Ionized Gas Emission Lines
    ion_lams = np.array([EMISSION_LINES['[FeII]'].value,
                         EMISSION_LINES['[SiVI]'].value,
                         EMISSION_LINES['[CaVIII]'].value,
                         EMISSION_LINES['Bry'].value,
                         EMISSION_LINES['Brd'].value,
                         EMISSION_LINES['HeI'].value])

    ion_names = ['[FeII]', '[SiVI]', '[CaVIII]', 'Bry', 'Brd', 'HeI']

    # Molecular Gas Emission Lines
    mol_lams = np.array([EMISSION_LINES['H2 (1-0) S(1)'].value,
                         EMISSION_LINES['H2 (1-0) S(0)'].value,
                         EMISSION_LINES['H2 (1-0) S(2)'].value,
                         EMISSION_LINES['H2 (1-0) S(3)'].value,
                         EMISSION_LINES['H2 (2-1) S(1)'].value,
                         EMISSION_LINES['H2 (2-1) S(0)'].value])
    mol_names = ['H2 (1-0) S(1)', 'H2 (1-0) S(0)', 'H2 (1-0) S(2)', 'H2 (1-0) S(3)', 'H2 (2-1) S(1)',
                 'H2 (2-1) S(0)']

    # Prominent Stellar lines that sometimes weren't corrected for in telluric subtraction
    stellar_lams = np.array([EMISSION_LINES['Br13'].value,
                             EMISSION_LINES['Br14'].value,
                             EMISSION_LINES['Br15'].value])
    stellar_names = ['Br13', 'Br14', 'Br15']

    # H-band Brackett lines
    hband_bry_lams = np.array([EMISSION_LINES['Br10'].value,
                               EMISSION_LINES['Br11'].value,
                               EMISSION_LINES['Br12'].value])

    hband_bry_names = ['Br10', 'Br11', 'Br12']

    flag = np.zeros_like(loglam, dtype=bool)

    if mask_ionized_lines:
        for name, lam in zip(ion_names, ion_lams):

            # Use 2000 km/s as the half width for masking broad Bry
            if (name == 'Bry') & mask_broad_bry:
                dvj = 2000.
            else:
                dvj = dv

            flag |= ((np.exp(loglam) > lam * (1 + z) * (1 - dvj / ckms)) &
                     (np.exp(loglam) < lam * (1 + z) * (1 + dvj / ckms)))

    if mask_h2_lines:
        for name, lam in zip(mol_names, mol_lams):
            flag |= ((np.exp(loglam) > lam * (1 + z) * (1 - dv / ckms)) &
                     (np.exp(loglam) < lam * (1 + z) * (1 + dv / ckms)))

    if mask_stellar_bry:
        for name, lam in zip(stellar_names, stellar_lams):
            flag |= ((np.exp(loglam) > lam * (1 - dv / ckms)) &
                     (np.exp(loglam) < lam * (1 + dv / ckms)))

    if mask_hband_bry:
        for name, lam in zip(hband_bry_names, hband_bry_lams):
            flag |= ((np.exp(loglam) > lam * (1 + z) * (1 - 2000. / ckms)) &
                     (np.exp(loglam) < lam * (1 + z) * (1 + 2000. / ckms)))

    # Always flag the beginning and the end of the stellar templates and noisy region between
    # H and K band
    flag |= np.exp(loglam) > lam_range_temp[1] * (1 + z) * (1 - 900 / ckms)
    flag |= np.exp(loglam) < lam_range_temp[0] * (1 + z) * (1 + 900 / ckms)
    flag |= (np.exp(loglam) > 1.8) & (np.exp(loglam) < 1.95)

    # Mask the first 5 and last 5 pixels if the spectrum has been smoothed
    # using ppxf_util.gaussian_filter1d
    if smooth:
        flag[0:6] = True
        flag[-5:] = True

    # Mask the regions of low atmospheric transmission in K-band
    if mask_telluric:
        flag |= (np.exp(loglam) > 1.995) & (np.exp(loglam) < 2.03)
        flag |= (np.exp(loglam) > 2.047) & (np.exp(loglam) < 2.078)

    return np.where(flag == 0)[0]


def setup_fit(loglam_gal, loglam_temp, z=0, dv_mask=800., velscale_ratio=1,
              mask_h2_lines=True, mask_ionized_lines=True,
              mask_broad_bry=False, mask_stellar_bry=False, mask_telluric=False,
              mask_hband_bry=False, velocity_guess=0., dispersion_guess=100., smooth=False):
    """
    Simple convenience function for gathering all of the necessary input for PPXF
    :param loglam_gal: Natural log of the galaxy wavelengths
    :param loglam_temp: Natural log of the template wavelengths
    :param z: Estimated redshift of the galaxy
    :param dv_mask: Half width to use in masking the emission lines
    :param velscale_ratio: Ratio between the velocity scale of the templates and the galaxy
    :param mask_h2_lines: True or False for masking H2 lines
    :param mask_ionized_lines: True of False for masking ionized lines
    :param mask_broad_bry: True or False for masking broad Bry
    :param mask_stellar_bry: True or False for masking residual stellar Bry lines
        from telluric correction
    :param velocity_guess: Guess for the rest frame stellar velocity in km/s
    :param dispersion_guess: Guess for the stellar velocity dispersion in km/s
    :return dv: Velocity shift between start of galaxy and template spectra
    :return goodpixels: List of galaxy pixels to use in fitting
    :return start: Starting guess of the velocity and dispersion
    """

    # Velocity "shift" between the starting wavelengths of the stellar templates and galaxy
    if velscale_ratio > 1:
        dv = (np.mean(loglam_temp[:velscale_ratio]) - loglam_gal[0]) * ckms
    else:
        dv = (loglam_temp[0] - loglam_gal[0]) * ckms

    # Setup up the vector of good pixels to be fit
    goodpixels = create_goodpixels(loglam_gal, [np.exp(loglam_temp[0]), np.exp(loglam_temp[-1])],
                                   z=z, dv=dv_mask, mask_h2_lines=mask_h2_lines,
                                   mask_ionized_lines=mask_ionized_lines,
                                   mask_broad_bry=mask_broad_bry, mask_stellar_bry=mask_stellar_bry,
                                   smooth=smooth, mask_telluric=mask_telluric,
                                   mask_hband_bry=mask_hband_bry)

    # Setup the beginning guess for the stellar velocity and dispersion
    # Assume 0 km/s velocity and user provided guess for the dispersion.
    start = [ckms * np.log(1 + z) + velocity_guess, dispersion_guess]

    return dv, goodpixels, start


def fit_single_spectrum(lam_gal, flux_gal, templates, velscale, start,
                        velscale_ratio=1, noise=None, goodpixels=None, dv=0,
                        add_poly_deg=4, smooth=False, smooth_sigma_pix=None,
                        clean=False, plot=False):
    """
    Fit a single spectrum with PPXF
    :param lam_gal:
    :param flux_gal:
    :param templates:
    :param velscale:
    :param start:
    :param velscale_ratio:
    :param noise:
    :param goodpixels:
    :param dv:
    :param add_poly_deg:
    :return:
    """

    # Smooth the spectrum to the resolution of the EMILES templates if requested
    # The smoothing length should be provided by the user
    if smooth:
        flux_gal = ppxf_util.gaussian_filter1d(flux_gal, smooth_sigma_pix)

    # Log rebin the spectrum to the given velocity scale
    flux_rebin_gal, log_lam_gal, velscale = ppxf_util.log_rebin([lam_gal[0], lam_gal[-1]],
                                                                flux_gal, velscale=velscale)

    # Generate a default goodpixels vector
    if goodpixels is None:
        goodpixels = np.arange(len(flux_rebin_gal))

    # Normalize the spectrum to avoid rounding error
    norm = np.nanmedian(flux_rebin_gal)
    flux_rebin_gal /= np.nanmedian(flux_rebin_gal)

    # Generate a noise spectrum if noise is None. Use 15% of the flux
    if noise is None:
        noise = 0.15*np.abs(flux_rebin_gal)
        noise[np.isnan(noise) | (noise == 0)] = 1.0

    # Remove NaNs and infs from goodpixels and change them to 0 in the spectrum
    nonfinite_pix = ~np.isfinite(flux_rebin_gal)
    flux_rebin_gal[nonfinite_pix] = 0.0

    for i in range(len(nonfinite_pix)):
        if nonfinite_pix[i]:
            test = np.where(goodpixels == i)[0]
            if len(test) > 0:
                goodpixels = np.delete(goodpixels, test[0])

    # Require at least 50% of the original pixels to do a fit
    if np.float(len(goodpixels))/np.float(len(log_lam_gal)) > 0.25:

        # Run ppxf
        pp = ppxf.ppxf(templates, flux_rebin_gal, noise, velscale, start, plot=False, moments=2,
                       degree=add_poly_deg, vsyst=dv, goodpixels=goodpixels,
                       velscale_ratio=velscale_ratio, clean=clean)


        vel = pp.sol[0]
        disp = pp.sol[1]
        chi2 = pp.chi2
        temp_weights = pp.weights
        gal = pp.galaxy
        bestfit = pp.bestfit
        stellar = pp.bestfit - pp.apoly
        apoly = pp.apoly
        residual = pp.galaxy - pp.bestfit

        if plot:

            pp.plot()
            fig = plt.gcf()

            return vel, disp, chi2, temp_weights, gal, bestfit, stellar, apoly, residual, norm, fig

        else:

            return vel, disp, chi2, temp_weights, gal, bestfit, stellar, apoly, residual, norm

    else:

        return 'Not enough pixels to fit.'


def fit_cube(cube, lam_gal, fwhm_gal, z=0, velscale=None, noise_cube=None, velscale_ratio=1,
             ages='all', metals='all', lam_range_min_temp=1.4, lam_range_max_temp=2.5,
             dv_mask=800., mask_h2_lines=True, mask_ionized_lines=True, mask_telluric=False,
             mask_broad_bry=False, mask_stellar_bry=False, mask_hband_bry=False,
             velocity_guess=0., dispersion_guess=100.,
             add_poly_deg=4, smooth=False, parallel=False, ncores=None):
    """
    Fit an entire IFU cube
    :param cube: data cube to be fit
    :param lam_gal: wavelength array of the data cube
    :param fwhm_gal: instrumental resolution of the data in km/s
    :param z: redshift of the data
    :param velscale: optional, velocity scale to rebin to
    :param noise_cube: optional, error cube
    :param velscale_ratio: optional, velocity scale ratio to determine the velocity scale to rebin the templates
    :param ages: optional, which stellar age templates to use. Default is 'all'
    :param metals: optional, which metallicity templates to use. Default is 'all'
    :param lam_range_min_temp: optional, minimum wavelength of the template spectra. Default is 1.4 micron.
    :param lam_range_max_temp: optional, maximum wavelength of the template spectra. Default is 2.5 micron.
    :param dv_mask: optional, Width of mask to apply to emission lines in km/s. Default is 800 km/s.
    :param mask_h2_lines: optional, Whether to mask all H2 emission lines. Default is True.
    :param mask_ionized_lines: optional, Whether to mask all ionized gas lines. Default is True.
    :param mask_telluric: optional, Whether to mask regions with strong telluric residuals. Default is False.
    :param mask_broad_bry: optional, Whether to mask a broad Bry component. Default is False.
    :param mask_stellar_bry: optional, Whether to mask Bry feature from telluric star. Default is False.
    :param mask_hband_bry: optional, Whether to mask H-band Bry emission lines. Default is False.
    :param velocity_guess: optional, Initial guess for the stellar velocity. Default is 0 km/s.
    :param dispersion_guess: optional, Initial guess for the stellar velocity dispersion. Default is 100 km/s.
    :param add_poly_deg: optional, Degree of additive polynomial. Default is 4.
    :param smooth: optional, Whether to smooth the data cube to the template resolution before fitting. Default is False.
    :param parallel: optional, Whether to use parallel processing. Default is False.
    :param ncores: optional, If using parallel processing, how many CPUS to use. Default is None.
    :return: result: Very large list of the fitting results of each spaxel
             waves: The rebinned wavelength array

    Notes
    -----
    Use construct_fit_products(result, cube.shape) to reconstruct the fit products into maps and cubes.
    """

    # Grab a single spectrum from the cube to setup the wavelength scale
    test_spec = cube[:, 0, 0]
    test_spec_new, loglam_gal, velscale = ppxf_util.log_rebin([lam_gal[0], lam_gal[-1]],
                                                              test_spec, velscale=velscale)

    # Setup the stellar template spectra
    lam_range_temp = [lam_range_min_temp, lam_range_max_temp]
    if not smooth:
        temps, loglam_temp = create_emiles_templates(lam_range_temp, fwhm_gal,
                                                     velscale/velscale_ratio,
                                                     ages, metals)
    else:
        temps, loglam_temp = create_emiles_templates(lam_range_temp, 2.35482*60.,
                                                     velscale/velscale_ratio,
                                                     ages, metals)

    # Setup the parameters for the PPXF fit
    dv, gp, start = setup_fit(loglam_gal, loglam_temp, z=z, dv_mask=dv_mask,
                              velscale_ratio=velscale_ratio,
                              mask_h2_lines=mask_h2_lines, mask_ionized_lines=mask_ionized_lines,
                              mask_broad_bry=mask_broad_bry, mask_stellar_bry=mask_stellar_bry,
                              mask_hband_bry=mask_hband_bry,
                              velocity_guess=velocity_guess, dispersion_guess=dispersion_guess,
                              smooth=smooth, mask_telluric=mask_telluric)

    # If smoothing the cube spectra, setup the smoothing lengths per pixel
    if smooth:
        fwhm_emiles = 60.*2.35482
        sigma_kms = np.sqrt(fwhm_emiles ** 2 - fwhm_gal ** 2) / 2.35482
        sigma_pix = sigma_kms / ckms * lam_gal / (lam_gal[1] - lam_gal[0])
    else:
        sigma_pix = None

    # Fit each of the spectra in the cube using parallel processing, if requested
    nrows = cube.shape[1]
    ncolumns = cube.shape[2]
    ind = np.indices((nrows, ncolumns))
    xx = ind[0].ravel()
    yy = ind[1].ravel()
    pixels = [(xx[i], yy[i]) for i in range(len(xx))]

    if parallel:
        import multiprocessing
        if ncores is None:
            ncores = multiprocessing.cpu_count()

        pool = multiprocessing.Pool(ncores)

        if noise_cube is not None:
            result = [(p, pool.apply_async(fit_single_spectrum, args=(lam_gal, cube[:, p[0], p[1]],
                      temps, None, start, velscale_ratio, noise_cube[:, p[0], p[1]],
                      gp, dv, add_poly_deg, smooth, sigma_pix)))
                      for p in pixels]

        else:
            result = [(p, pool.apply_async(fit_single_spectrum, args=(lam_gal, cube[:, p[0], p[1]],
                      temps, None, start, velscale_ratio, None,
                      gp, dv, add_poly_deg, smooth, sigma_pix))) for p in pixels]

        result = [(p[0], p[1].get()) for p in result]

        pool.close()

    else:

        if noise_cube is not None:
            result = [(p, fit_single_spectrum(lam_gal, cube[:, p[0], p[1]], temps, None, start,
                                              velscale_ratio, noise_cube[:, p[0], p[1]], gp, dv,
                                              add_poly_deg, smooth, sigma_pix)) for p in pixels]

        else:
            result = [(p, fit_single_spectrum(lam_gal, cube[:, p[0], p[1]], temps, None, start,
                      velscale_ratio, None, gp, dv, add_poly_deg, smooth, sigma_pix))
                      for p in pixels]

    return result, np.exp(loglam_gal)


def construct_fit_products(result, cube_shape):

    vel = np.zeros((cube_shape[1], cube_shape[2]))
    disp = np.zeros((cube_shape[1], cube_shape[2]))
    chi2 = np.zeros((cube_shape[1], cube_shape[2]))
    flag = np.ones((cube_shape[1], cube_shape[2]))
    norm = np.zeros((cube_shape[1], cube_shape[2]))

    cube_gal = np.zeros(cube_shape)
    cube_bestfit = np.zeros(cube_shape)
    cube_stellar = np.zeros(cube_shape)
    cube_apoly = np.zeros(cube_shape)
    cube_residual = np.zeros(cube_shape)

    test = False

    for r in result:
        i = r[0][0]
        j = r[0][1]

        if r[1] == 'Not enough pixels to fit.':

            flag[i, j] = 0

        else:

            fit_data = r[1]

            if not test:
                temp_weights = np.zeros((len(fit_data[3]), cube_shape[1], cube_shape[2]))
                test = True

            vel[i, j] = fit_data[0]
            disp[i, j] = fit_data[1]
            chi2[i, j] = fit_data[2]
            temp_weights[:, i, j] = fit_data[3]
            cube_gal[:, i, j] = fit_data[4]
            cube_bestfit[:, i, j] = fit_data[5]
            cube_stellar[:, i, j] = fit_data[6]
            cube_apoly[:, i, j] = fit_data[7]
            cube_residual[:, i, j] = fit_data[8]
            norm[i, j] = fit_data[9]

    return (vel, disp, chi2, temp_weights, cube_gal, cube_bestfit, cube_stellar,
            cube_apoly, cube_residual, norm, flag)


def from_pkl_to_fits(file, crpix1, crpix2, crval1, crval2, cdelt, object, save_dir='./'):
    """
    Write out FITS files for all of the arrays contained in the PPXF pickle files.
    :param file: String with the pathname to the pickle file.
    :return:
    """

    f = open(file, 'rb')
    p = pickle.load(f)

    products = p[0]
    lam_rebin = p[1]

    vel = products[0]
    disp = products[1]
    chi2 = products[2]
    temp_weights = products[3]
    cube_gal = products[4]
    cube_bestfit = products[5]
    cube_stellar = products[6]
    cube_apoly = products[7]
    cube_residual = products[8]
    norm = products[9]
    flag = products[10]

    nspec = cube_gal.shape[0]

    # Multiply the cubes by the normalization that was divided out during the fitting
    for i in range(nspec):
        cube_gal[i, :, :] = cube_gal[i, :, :] * norm
        cube_bestfit[i, :, :] = cube_bestfit[i, :, :] * norm
        cube_stellar[i, :, :] = cube_stellar[i, :, :] * norm
        cube_apoly[i, :, :] = cube_apoly[i, :, :] * norm
        cube_residual[i, :, :] = cube_residual[i, :, :] * norm

    # Create the HDUs for each product
    vel_hdu = fits.PrimaryHDU(data=vel)
    disp_hdu = fits.PrimaryHDU(data=disp)
    chi2_hdu = fits.PrimaryHDU(data=chi2)
    temp_weights_hdu = fits.PrimaryHDU(data=temp_weights)
    cube_gal_hdu = fits.PrimaryHDU(data=cube_gal)
    cube_bestfit_hdu = fits.PrimaryHDU(data=cube_bestfit)
    cube_stellar_hdu = fits.PrimaryHDU(data=cube_stellar)
    cube_apoly_hdu = fits.PrimaryHDU(data=cube_apoly)
    cube_residual_hdu = fits.PrimaryHDU(data=cube_residual)
    flag_hdu = fits.PrimaryHDU(data=flag)

    # Populate the keywords for the cubes
    for h in [cube_gal_hdu, cube_bestfit_hdu, cube_stellar_hdu, cube_apoly_hdu,
              cube_residual_hdu]:
        h.header['CTYPE3'] = 'WAVE'
        h.header['CRPIX3'] = 1
        h.header['CRVAL3'] = np.log(lam_rebin[0])
        h.header['CDELT3'] = np.log(lam_rebin[1]) - np.log(lam_rebin[0])
        h.header['CUNIT3'] = 'LOG(MICRON)'

    # Populate the header keywords
    for h in [vel_hdu, disp_hdu, chi2_hdu, temp_weights_hdu, cube_gal_hdu,
              cube_bestfit_hdu, cube_stellar_hdu, cube_apoly_hdu, cube_residual_hdu,
              flag_hdu]:

        h.header['CRPIX1'] = crpix1
        h.header['CRPIX2'] = crpix2
        h.header['CRVAL1'] = crval1
        h.header['CRVAL2'] = crval2
        h.header['CDELT1'] = -cdelt
        h.header['CDELT2'] = cdelt
        h.header['CD1_1'] = -cdelt
        h.header['CD1_2'] = 0.0
        h.header['CD2_1'] = 0.0
        h.header['CD2_2'] = cdelt
        h.header['CTYPE1'] = 'RA---TAN'
        h.header['CTYPE2'] = 'DEC--TAN'
        h.header['CUNIT1'] = 'DEGREE'
        h.header['CUNIT2'] = 'DEGREE'
        h.header['RADECSYS'] = 'FK5'
        h.header['EQUINOX'] = 2000.
        h.header['OBJECT'] = object

    vel_hdu.writeto(save_dir+'stellar_velocity_ppxf.fits', overwrite=True)
    disp_hdu.writeto(save_dir+'stellar_dispersion_ppxf.fits', overwrite=True)
    chi2_hdu.writeto(save_dir+'chi-square_ppxf.fits', overwrite=True)
    temp_weights_hdu.writeto(save_dir+'template_weights_ppxf.fits', overwrite=True)
    cube_gal_hdu.writeto(save_dir+'galaxy_cube_ppxf.fits', overwrite=True)
    cube_bestfit_hdu.writeto(save_dir + 'bestfit_total_cube_ppxf.fits', overwrite=True)
    cube_stellar_hdu.writeto(save_dir + 'bestfit_stellar_cube_ppxf.fits', overwrite=True)
    cube_apoly_hdu.writeto(save_dir + 'bestfit_apoly_cube_ppxf.fits', overwrite=True)
    cube_residual_hdu.writeto(save_dir + 'residual_cube_ppxf.fits', overwrite=True)
    flag_hdu.writeto(save_dir + 'flag_ppxf.fits', overwrite=True)

