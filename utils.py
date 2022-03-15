#!/usr/bin/env python3
import pyrtools as pt
import scipy
import numpy as np
from scipy import fft as sp_fft


def amplitude_spectra(image):
    """Compute amplitude spectra of an image.

    We compute the 2d Fourier transform of an image, take its magnitude, and
    then radially average it. This averages across orientations and also
    discretizes the frequency. We also drop a disk in frequency space to
    exclude the highest frequencies (that is, those where we don't have
    cardinal directions).

    Parameters
    ----------
    image : np.ndarray
        The 2d array containing the image

    Returns
    -------
    spectra : np.ndarray
        The 1d array containing the amplitude spectra

    Notes
    -----
    See
    https://scipy-lectures.org/advanced/image_processing/auto_examples/plot_radial_mean.html
    for how we compute the radial mean. Note the tutorial excludes label=0, but
    we include it (corresponds to the DC term).

    """
    frq = sp_fft.fftshift(sp_fft.fft2(image))
    # following
    # https://scipy-lectures.org/advanced/image_processing/auto_examples/plot_radial_mean.html.
    # Note the tutorial excludes label=0, but we include it (corresponds to the
    # DC term).
    rbin = pt.synthetic_images.polar_radius(frq.shape).astype(np.int)
    # we ignore all frequencies outside a disk centered at the origin that
    # reaches to the first edge (in frequency space). This means we get all
    # frequencies that we can measure in each orientation (you can't get any
    # frequencies in the cardinal directions beyond this disk)
    frq_disk = pt.synthetic_images.polar_radius(frq.shape)
    frq_thresh = min(frq.shape)//2
    frq_disk = frq_disk < frq_thresh
    rbin[~frq_disk] = rbin.max()+1
    spectra = scipy.ndimage.mean(np.abs(frq), labels=rbin,
                                 index=np.arange(frq_thresh-1))
    return spectra


def _construct_impulse_pyr(size, height, order, is_complex=False):
    """Construct pyramid with impulse in each band."""
    empty_image = np.zeros((size, size))
    pyr = pt.pyramids.SteerablePyramidFreq(empty_image, height=height, order=order,
                                           is_complex=is_complex)

    # Put an impulse into the middle of each band
    for k, v in pyr.pyr_size.items():
        mid = (v[0]//2, v[1]//2)
        pyr.pyr_coeffs[k][mid] = 1

    return pyr


def get_steerpyr_filters(size, height='auto', order=1, is_complex=False):
    """Construct and return steerpyr filters.

    We do this by getting the impulse response of the SteerablePyramidFreq and
    then reconstructing each band.

    Parameters
    ----------
    size : int
        Size of the image to build the filter on.
    height : 'auto' or int, optional
        Number of scales to build. If 'auto', calculate based on size.
    order : int
        Order of steerable pyramid filters.
    is_complex : bool
        Whether the coefficients are complex- or real-valued.

    Returns
    -------
    filters: dict
        Dictionary of filters

    """
    pyr = _construct_impulse_pyr(size, height, order, is_complex)

    # And take a look at the reconstruction of each band:
    filters = {}
    for k in pyr.pyr_coeffs.keys():
        if isinstance(k, tuple):
            filters[k] = pyr.recon_pyr(*k)
    for k in ['residual_highpass', 'residual_lowpass']:
        filters[k] = pyr.recon_pyr(k)

    return filters


def get_steerpyr_freq_filters(size, height='auto', order=1, is_complex=False):
    """Construct and return steerpyr filters in frequency domain.

    We do this by getting the impulse response of the SteerablePyramidFreq and
    then reconstructing each band.

    Parameters
    ----------
    size : int
        Size of the image to build the filter on.
    height : 'auto' or int, optional
        Number of scales to build. If 'auto', calculate based on size.
    order : int
        Order of steerable pyramid filters.
    is_complex : bool
        Whether the coefficients are complex- or real-valued.

    Returns
    -------
    filters: dict
        Dictionary of frequency domain filters

    """
    pyr = _construct_impulse_pyr(size, height, order, is_complex)
    filters = {}
    for k in pyr.pyr_coeffs.keys():
        if isinstance(k, tuple):
            basisFn = pyr.recon_pyr(*k)
            basisFmag = np.fft.fftshift(np.abs(np.fft.fft2(basisFn, (size, size))))
            filters[k] = basisFmag

    for k in ['residual_highpass', 'residual_lowpass']:
        basisFn = pyr.recon_pyr(k)
        basisFmag = np.fft.fftshift(np.abs(np.fft.fft2(basisFn, (size, size))))
        filters[k] = basisFmag

    return filters
