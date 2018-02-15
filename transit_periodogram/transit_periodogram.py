# -*- coding: utf-8 -*-

from __future__ import division, print_function

import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, *args, **kwargs: x

from .transit_periodogram_impl import transit_periodogram_impl


__all__ = ["transit_periodogram"]


def transit_periodogram(time, flux, periods, durations, flux_err=None,
                        oversample=10, method=None):
    """Compute the transit periodogram for a light curve.

    Parameters
    ----------
    time : array-like
        Array with time measurements
    flux : array-like
        Array of measured fluxes at ``time``
    periods : array-like
        The periods to search in the same units as ``time``
    durations : array-like
        The transit durations to consider in the same units
        as ``time``
    flux_err : array-like
        The uncertainties on ``flux``
    method :
        The periodogram type. Allowed values (a) ``snr`` to select on depth
        signal-to-noise ratio, (b) ``likelihood`` the log-likelihood of the
        model

    Returns
    -------
    periods : array-like
        Set of trial periods
    objective : array-like
        Either depth signal-to-noise ratio or the loglikelihood evaluated at the
        maximum likelihood estimate of the depth as a function of period
        depending on ``method``
    log_like : array-like
        Loglikelihood evaluated at the maximum likelihood estimate of the depth
        as a function of period
    depth_snr : array-like
        Depth signal-to-noise ratio as a function of period
    depth : array-like
        Maximum likelihood estimate for the transit depth as a function of
        period
    depth_errs : array-like
        Uncertainties (one standard deviation) associated with the maximum
        likelihood depth as a function of period
    phase : array-like
        Mid-transit time estimate as a function of period
    duration : array-like
        Transit duration estimate as a function of period
    """
    try:
        oversample = int(oversample)
    except TypeError:
        raise ValueError("oversample must be an integer,"
                         " got {0}".format(oversample))

    if method is None:
        method = "likelihood"
    allowed_methods = ["snr", "likelihood"]
    if method not in allowed_methods:
        raise ValueError("unrecognized method '{0}'\nallowed methods are: {1}"
                         .format(method, allowed_methods))

    time = np.ascontiguousarray(time, dtype=np.float64)
    flux = np.ascontiguousarray(flux, dtype=np.float64)

    if time.shape != flux.shape:
        raise ValueError("time and flux arrays have different shapes: "
                         "{0}, {1}".format(time.shape, flux.shape))
    if flux_err is None:
        flux_err = np.ones_like(flux)

    flux_err = np.ascontiguousarray(flux_err, dtype=np.float64)
    flux_ivar = 1.0 / flux_err**2

    periods = np.atleast_1d(periods)
    use_likelihood = (method == "likelihood")
    durations = np.ascontiguousarray(np.atleast_1d(np.abs(durations)),
                                     dtype=np.float64)

    return transit_periodogram_impl(time, flux, flux_ivar, periods, durations,
                                    oversample, use_likelihood=use_likelihood)
