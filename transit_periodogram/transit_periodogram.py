# -*- coding: utf-8 -*-

from __future__ import division, print_function

import numpy as np

from .transit_periodogram_impl import transit_periodogram_impl


__all__ = ["transit_periodogram"]


def transit_periodogram(time, flux, periods, durations, flux_err=None,
                        oversample=10, method=None):
    """Compute the transit periodogram for a light curve.

    Parameters
    ----------
    time : array-like
        An array of timestamps
    flux : array-like
        The fluxes measured at each time in the ``time`` array
    periods : array-like
        The periods to search in the same units as ``time``
    durations : array-like
        The transit durations to consider in the same units as ``time``
    flux_err : array-like
        (optional) The uncertainties on ``flux``. If these are not provided
        then the uncertainties estimated for the depth shouldn't be taken
        seriously.
    method :
        The periodogram type. Allowed values (a) ``snr`` to select on depth
        signal-to-noise ratio, (b) ``likelihood`` the log-likelihood of the
        model

    Returns
    -------
    periodogram : array-like
        Either depth signal-to-noise ratio or the log likelihood evaluated at
        maximum (across duration and phase) depending on ``method``
    depth : array-like
        The best fit estimate of the transit depth where "best fit" is defined
        by ``method``
    depth_err : array-like
        The 1-sigma uncertainties on the depth estimates in ``depth``
    phase : array-like
        The best fit mid-transit time at each period
    duration : array-like
        The best fit transit duration at each period
    depth_snr : array-like
        The signal-to-noise of the depth measurement (this is equal to
        ``depth/depth_err``)
    log_like : array-like
        The log likelihood of the model at each point in the periodogram

    """
    # Check for absurdities in the ``oversample`` choice
    try:
        oversample = int(oversample)
    except TypeError:
        raise ValueError("oversample must be an integer,"
                         " got {0}".format(oversample))
    if oversample < 1:
        raise ValueError("oversample must be greater than or equal to 1")

    # Format an check the input period and duration arrays
    periods = np.atleast_1d(periods)
    use_likelihood = (method == "likelihood")
    durations = np.ascontiguousarray(np.atleast_1d(np.abs(durations)),
                                     dtype=np.float64)
    if np.max(durations) >= np.min(periods):
        raise ValueError("the maximum transit duration must be shorter than "
                         "the minimum period")

    # Select the periodogram type
    if method is None:
        method = "likelihood"
    allowed_methods = ["snr", "likelihood"]
    if method not in allowed_methods:
        raise ValueError("unrecognized method '{0}'\nallowed methods are: {1}"
                         .format(method, allowed_methods))

    # Format and check the input arrays
    time = np.ascontiguousarray(time, dtype=np.float64)
    flux = np.ascontiguousarray(flux, dtype=np.float64)
    if time.shape != flux.shape:
        raise ValueError("time and flux arrays have different shapes: "
                         "{0}, {1}".format(time.shape, flux.shape))

    # If an array of uncertainties is not provided, estimate it using the MAD
    if flux_err is None:
        flux_err = np.median(np.abs(np.diff(flux))) + np.zeros_like(flux)
    flux_err = np.ascontiguousarray(flux_err, dtype=np.float64)
    flux_ivar = 1.0 / flux_err**2

    # Normalize the flux array for numerics
    flux_norm = flux - np.median(flux)

    # Run the real code
    return transit_periodogram_impl(time, flux_norm, flux_ivar, periods,
                                    durations, oversample,
                                    use_likelihood=use_likelihood)
