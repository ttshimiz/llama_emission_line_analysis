# Useful functions for the analysis

import numpy as np
import astropy.units as u
import astropy.constants as c
import astropy.io.fits as fits
import astropy.modeling as apy_mod
from astropy.stats import sigma_clipped_stats
import aplpy
from spectral_cube import SpectralCube
from astropy.wcs import WCS
import scipy.ndimage as scp_ndi
import matplotlib.pyplot as plt


def read_cube(fn, scale=True):
    """
    Reads in SINFONI FITS cube, cleans up the header, and returns a
    spectral-cube object.

    Parameters
    ----------
    fn = string; FITS file names

    Returns
    -------
    cube = spectral_cube.SpectralCube object

    """

    if scale:
        data = fits.getdata(fn) * 1e-17
    else:
        data = fits.getdata(fn)

    header = fits.getheader(fn)

    # Check the spectral axis units and values
    naxis3 = header['NAXIS3']
    cunit3 = header['CUNIT3']
    crval3 = header['CRVAL3']
    cdelt3 = header['CDELT3']
    crpix3 = header['CRPIX3']
    lam = (np.arange(naxis3) + 1 - crpix3) * cdelt3 + crval3

    if cunit3 == 'LOG(MICRON)':
        lam = np.exp(lam)

    return data, lam, header


def create_rgb_image(rfile, gfile, bfile, scale=1e21, stretch=12, Q=0.1,
                     sn_cut=3.0):
    """
    Function to create an RGB image from 3 emission line flux maps.
    Uses astropy.visualization.make_lupton_rgb to generate the RGB values.
    Uses aplpy to create and show the figure.

    Parameters
    ----------
    rfile/gfile/bfile = strings for the FITS files containing the line maps corresponding
                        to each color
    scale = float or 3 element array with values to multiply the integrated fluxes by.
    stretch = float for the linear stretch of the image
    Q = float for the asinh softening parameter
    sn_cut = S/N threshold of each line, all pixels with S/N < sn_cut will be set to 0

    For more information on stretch and Q see the documentation for make_lupton_rgb and
    Lupton (2004).

    Returns
    -------
    fig = aplpy.FITSFigure object

    """

    from astropy.visualization import make_lupton_rgb

    hdu_r = fits.open(rfile)
    hdu_g = fits.open(gfile)
    hdu_b = fits.open(bfile)

    image_r = hdu_r['int flux'].data
    image_r_err = hdu_r['int flux error'].data

    image_g = hdu_g['int flux'].data
    image_g_err = hdu_g['int flux error'].data

    image_b = hdu_b['int flux'].data
    image_b_err = hdu_b['int flux error'].data

    ind_r = (image_r / image_r_err < sn_cut) | np.isnan(image_r)
    ind_g = (image_g / image_g_err < sn_cut) | np.isnan(image_g)
    ind_b = (image_b / image_b_err < sn_cut) | np.isnan(image_b)

    image_r[ind_r] = 0
    image_g[ind_g] = 0
    image_b[ind_b] = 0

    if np.isscalar(scale):
        image_r = image_r * scale
        image_g = image_g * scale
        image_b = image_b * scale
    else:
        image_r = image_r * scale[0]
        image_g = image_g * scale[1]
        image_b = image_b * scale[2]

    rgb = make_lupton_rgb(image_r, image_g, image_b, filename='rgb.png', Q=Q, stretch=stretch)

    hdu_r[0].header['CTYPE1'] = 'RA---TAN'
    hdu_r[0].header['CTYPE2'] = 'DEC--TAN'

    fig = aplpy.FITSFigure(hdu_r)
    fig.show_rgb('rgb.png', interpolation='gaussian')

    return fig


def find_cont_center(cube, lam, lamrange, guess=None, plot=False, header=None):
    """
    Function to fit a 2D Gaussian to the image of a user-defined continuum
    """

    slice = (lam > lamrange[0]) & (lam < lamrange[1])
    int = np.sum(cube[slice, :, :], axis=0)
    img = int / np.nanmean(int)
    xx, yy = np.meshgrid(range(img.shape[1]), range(img.shape[0]))

    if guess is None:
        guess_x = img.shape[1] / 2
        guess_y = img.shape[0] / 2
    else:
        guess_x = guess[0]
        guess_y = guess[1]
    img_cut = img[guess_y - 10:guess_y + 10, guess_x - 10:guess_x + 10]
    xx_cut = xx[guess_y - 10:guess_y + 10, guess_x - 10:guess_x + 10]
    yy_cut = yy[guess_y - 10:guess_y + 10, guess_x - 10:guess_x + 10]
    gauss_mod = apy_mod.models.Gaussian2D(x_mean=guess_x, y_mean=guess_y,
                                          x_stddev=3.0, y_stddev=3.0)
    fitter = apy_mod.fitting.LevMarLSQFitter()

    best_fit = fitter(gauss_mod, xx_cut, yy_cut, img_cut)

    center = [best_fit.x_mean.value, best_fit.y_mean.value]

    if plot:

        hdu = fits.PrimaryHDU(data=int, header=header)
        fig = aplpy.FITSFigure(hdu)
        fig.show_colorscale(cmap='cubehelix', stretch='linear')
        ra, dec = fig.pixel2world(center[0] + 1, center[1] + 1)
        fig.show_markers(ra, dec, marker='+', c='k', s=100, lw=1.0)
        fig.add_colorbar()
        fig.add_label(0.05, 0.95,
                      'Continuum = {0:0.3f} - {1:0.3f} micron'.format(lamrange[0], lamrange[1]),
                      relative=True, color='r', size=14, horizontalalignment='left')
        fig.add_label(0.05, 0.90,
                      'Pixel = [{0:0.2f},{1:0.2f}]'.format(center[0], center[1]),
                      relative=True, color='r', size=14, horizontalalignment='left')
        fig.add_label(0.05, 0.85,
                      'RA, DEC = [{0:0.4f},{1:0.4f}]'.format(ra, dec),
                      relative=True, color='r', size=14, horizontalalignment='left')

        return center, best_fit, fig

    else:

        return center, best_fit


def fitpa(vel, vel_err=None, xoff=None, yoff=None, mask=None, border=True, debug=False):
    """
    Function to fit the kinematic position angle using fit_kinematic_pa from Michele Cappelari
    """
    import sys
    sys.path.append('/Users/ttshimiz/Github/fit_kinematic_pa/')
    sys.path.append('/Users/ttshimiz/Github/display_pixels/')
    from fit_kinematic_pa import fit_kinematic_pa

    x, y = np.meshgrid(range(vel.shape[1]), range(vel.shape[0]))
    x = x - vel.shape[0] / 2
    y = y - vel.shape[1] / 2

    if vel_err is None:
        vel_err = vel * 0 + 10.

    if mask is None:
        mask = np.isnan(vel)

    xx = x.flatten()
    yy = y.flatten()
    vel_flat = vel.flatten()
    vel_err_flat = vel_err.flatten()
    mask_flat = mask.flatten()

    if border:
        border_mask = (np.abs(xx) > 10) | (np.abs(yy) > 10)
        mask_all = (~mask_flat) & (~border_mask)
    else:
        mask_all = (~mask_flat)

    if xoff is not None:
        xx = xx - xoff + vel.shape[0] / 2

    if yoff is not None:
        yy = yy - yoff + vel.shape[1] / 2

    xx = xx[mask_all]
    yy = yy[mask_all]
    vel_flat = vel_flat[mask_all]
    vel_err_flat = vel_err_flat[mask_all]

    angBest, angErr, vSyst, fig = fit_kinematic_pa(xx, yy, vel_flat, dvel=vel_err_flat, debug=debug)

    return angBest, angErr, vSyst, fig


def calc_pv_diagram(cube, slit_width, slit_angle, pxs, soff=0.):
    """
    Calculate a PV diagram from a cube
    :param cube: The data cube
    :param slit_width: Width of the slit to use in arcseconds
    :param slit_angle: Angle of the slit east of north in degrees
    :param pxs: Pixel scale of the cube in arcseconds
    :param soff: Vertical offset of the slit
    :return: pv: 2D array with positional offset as rows and velocity as columns
    """

    cube_shape = cube.shape
    psize = cube_shape[1]
    vsize = cube_shape[0]
    lin = np.arange(psize) - np.fix(psize / 2.)

    # The angle that is needed for Scipy's rotation should be north of east
    rotate_angle = slit_angle - 180.
    veldata = scp_ndi.interpolation.rotate(cube, rotate_angle, axes=(2, 1),
                                           reshape=False)
    tmpn = (((lin * pxs) <= (soff + slit_width / 2.)) &
            ((lin * pxs) >= (soff - slit_width / 2.)))
    data = np.zeros((psize, vsize))

    for i in range(psize):
        for j in range(vsize):
            data[i, j] = np.nansum(veldata[j, i, tmpn])

    return data


def measure_1d_profile(cube, slit_width, slit_angle, pxs, vx, soff=0.):
    """
    Measure the rotation curve of an emission line along a specific slit
    :param cube: Data cube where the rotation curve will be measured
    :param slit_width: Width of the slit which will define the apertures in arcseconds
    :param slit_angle: Position angle of the slit in degrees East of North
    :param pxs: Pixelscale
    :param vx: Velocity axis for the cube
    :param soff: Offset from center of the slit perpendicular to slit position angle
    :return: 1D arrays of the position, flux, velocity, and dispersion calculated along the slit
    """

    # First convert the cube to a PV array
    pv = calc_pv_diagram(cube, slit_width, slit_angle, pxs, soff=soff)

    # Bin the data further into single spectra by summing rows that are width of the slit
    # This effectively creates a summed spectrum within a rectangular aperture in the original cube.
    nrows = pv.shape[0]
    nbins = np.int(np.ceil((nrows * pxs) / slit_width))
    offset = np.zeros(nbins)
    flux = np.zeros(nbins)
    vel = np.zeros(nbins)
    disp = np.zeros(nbins)

    for i in range(nbins):

        if i == (nbins - 1):

            bin_start = np.int(i * np.round(slit_width / pxs))
            bin_end = nrows + 1

        else:

            bin_end = np.int((i + 1) * np.round(slit_width / pxs))
            bin_start = np.int(i * np.round(slit_width / pxs))

        spec = np.nansum(pv[bin_start:bin_end, :], axis=0)

        # Use the first and second moment as a guess of the line parameters
        mom0 = np.sum(spec)
        mom1 = np.sum(spec * vx)/mom0
        mom2 = np.sum(spec * (vx - mom1)**2)/mom0

        mod = apy_mod.models.Gaussian1D(amplitude=mom0/np.sqrt(2*np.pi*np.abs(mom2)), mean=mom1,
                                        stddev=np.sqrt(np.abs(mom2)))
        mod.amplitude.bounds = (0, None)
        mod.stddev.bounds = (0, None)
        fitter = apy_mod.fitting.LevMarLSQFitter()
        best_fit = fitter(mod, vx, spec)

        plt.figure()
        plt.plot(vx, spec)
        plt.plot(vx, best_fit(vx))

        vel[i] = best_fit.mean.value
        disp[i] = best_fit.stddev.value
        flux[i] = best_fit.amplitude.value*np.sqrt(2*np.pi)*disp[i]
        offset[i] = ((bin_end + bin_start) - (nrows))*pxs/2

    return offset, flux, vel, disp
