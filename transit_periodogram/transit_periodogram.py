# -*- coding: utf-8 -*-

from __future__ import division, print_function

import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, *args, **kwargs: x

from .transit_periodogram_impl import fold as _fold


__all__ = ["transit_periodogram"]


def transit_periodogram(time, flux, periods, durations, flux_err=None,
                        oversample=10, progress=False, method=None):
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
        Trial periods
    depth_snr : array-like
        Signal-to-noise ratio evaluated at the maximum likelihood depth
        for each period
    depths : array-like
        Maximum likelihood depths for each period
    depth_errs : array-like
        Uncertainties (1-sigma) associated with the maximum likelihood depth
        for each period
    phase : array-like
        Mid-transit times

    """
    try:
        oversample = int(oversample)
    except TypeError:
        raise ValueError("oversample must be an integer,"
                         " got {0}".format(oversample))

    if method is None:
        method = "snr"
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

    sum_flux2_all = np.sum(flux * flux * flux_ivar)
    sum_flux_all = np.sum(flux * flux_ivar)
    sum_ivar_all = np.sum(flux_ivar)

    periods = np.atleast_1d(periods)
    periodogram = -np.inf + np.zeros_like(periods)
    log_likelihood = np.empty_like(periods)
    depth_snr = np.empty_like(periods)
    depths = np.empty_like(periods)
    depth_errs = np.empty_like(periods)
    phases = np.empty_like(periods)
    best_durations = np.empty_like(periods)

    gen = periods
    if progress:
        gen = tqdm(periods, total=len(periods))

    use_likelihood = (method == "likelihood")
    durations = np.ascontiguousarray(np.atleast_1d(np.abs(durations)), dtype=np.float64)
    for i, period in enumerate(gen):
        depth, depth_var, ll, phase, duration = _fold(time, flux, flux_ivar,
                                                      sum_flux2_all, sum_flux_all,
                                                      sum_ivar_all, period,
                                                      durations, oversample,
                                                      use_likelihood=use_likelihood)
        snr = depth / np.sqrt(depth_var)
        if use_likelihood:
            objective = ll
        else:
            objective = snr

        periodogram[i] = objective
        log_likelihood[i] = ll
        depth_snr[i] = snr
        depths[i] = depth
        depth_errs[i] = np.sqrt(depth_var)
        phases[i] = phase
        best_durations[i] = duration

    return (
        periods, periodogram, log_likelihood, depth_snr, depths, depth_errs,
        phases, best_durations
    )
