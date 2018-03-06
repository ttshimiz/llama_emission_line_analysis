"""
Michele Cappellari, Oxford, 8 February 2018

"""

import numpy as np
from scipy import fftpack

from ppxf import losvd_rfft, rebin
import miles_util as lib

################################################################################

def convolve_gauss_hermite(templates, start, velscale, npix,
                           velscale_ratio=1, sigma_diff=0, vsyst=0):
    """
    Convolve a spectrum, or a set of spectra, arranged into columns of an array,
    with a LOSVD parametrized by the Gauss-Hermite series.
    This is intended to reproduce what pPXF does for the convolution.

    EXAMPLE:

        pp = ppxf(templates, galaxy, noise, velscale, start,
                  vsyst=dv, velscale_ratio=velscale_ratio)

        templates = templates @ pp.weights

        spec = convolve_gauss_hermite(templates, pp.sol, velscale, galaxy.size,
                                      velscale_ratio=velscale_ratio, vsyst=dv)

        # The spectrum below is *identical* to pp.bestfit to machine accuracy

        spectrum = spec*pp.mpoly + pp.apoly

    :param spectra: log rebinned spectra
    :param start: parameters of the LOSVD [vel, sig, h3, h4,...]
    :param velscale: velocity scale c*dLogLam in km/s
    :param npix: number of output pixels
    :return: vector or array with convolved spectra
    """
    templates = templates.reshape(templates.shape[0], -1)
    npix_temp, ntemp = templates.shape
    start = np.array(start)  # make copy
    start[:2] /= velscale
    vsyst /= velscale
    npad = fftpack.next_fast_len(npix_temp)

    templates_rfft = np.fft.rfft(templates, npad, axis=0)
    lvd_rfft = losvd_rfft(start, 1, start.shape, templates_rfft.shape[0],
                          1, vsyst, velscale_ratio, sigma_diff)

    # conv_temp = np.empty((npix, ntemp))
    # for j, template_rfft in enumerate(templates_rfft.T):  # columns loop
    #     tt = np.fft.irfft(template_rfft*lvd_rfft[:, 0, 0], npad)
    #     conv_temp[:, j] = rebin(tt[:npix*velscale_ratio], velscale_ratio)

    tt = np.fft.irfft(templates_rfft*lvd_rfft[:, 0], npad, axis=0)
    conv_temp = rebin(tt[:npix*velscale_ratio, :], velscale_ratio)

    return np.squeeze(conv_temp)

################################################################################

if __name__ == '__main__':

    import matplotlib.pyplot as plt
    import miles_util as miles

    FWHM_gal = 0.
    velscale = 70.
    pathname = 'public_programs/ppxf/miles_models/Mun1.30*.fits'
    miles = lib.miles(pathname, velscale, FWHM_gal, FWHM_tem=0)
    start = [2000, 1000, 0.1, 0.1]

    conv_temp = convolve_gauss_hermite(miles.templates, start, velscale, npix=2000, vsyst=-1e4)
    plt.plot(conv_temp)

    plt.pause(1)



