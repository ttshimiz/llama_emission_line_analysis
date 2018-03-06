# Useful functions for the analysis

import numpy as np
import astropy.units as u
import astropy.constants as c
import astropy.io.fits as fits
from astropy.stats import sigma_clipped_stats
import aplpy
from spectral_cube import SpectralCube
from astropy.wcs import WCS


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
        data = fits.getdata(fn)*1e-17
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

    return data, lam


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

    ind_r = (image_r/image_r_err < sn_cut) | np.isnan(image_r)
    ind_g = (image_g/image_g_err < sn_cut) | np.isnan(image_g)
    ind_b = (image_b/image_b_err < sn_cut) | np.isnan(image_b)

    image_r[ind_r] = 0
    image_g[ind_g] = 0
    image_b[ind_b] = 0

    if np.isscalar(scale):
        image_r = image_r*scale
        image_g = image_g*scale
        image_b = image_b*scale
    else:
        image_r = image_r*scale[0]
        image_g = image_g*scale[1]
        image_b = image_b*scale[2]

    rgb = make_lupton_rgb(image_r, image_g, image_b, filename='rgb.png', Q=Q, stretch=stretch)

    fig = aplpy.FITSFigure(hdu_r)
    fig.show_rgb('rgb.png', interpolation='gaussian')

    return fig
