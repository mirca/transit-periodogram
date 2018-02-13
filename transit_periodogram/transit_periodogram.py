# -*- coding: utf-8 -*-

from __future__ import division, print_function

import logging
import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, *args, **kwargs: x

try:
    from fast_histogram import histogram1d
except ImportError:
    HAS_FAST_HISTOGRAM = False
    histogram = np.histogram
else:
    HAS_FAST_HISTOGRAM = True
    def histogram(*args, **kwargs):
        return histogram1d(*args, **kwargs), None

from .transit_periodogram_impl import fold as _fold


__all__ = ["transit_periodogram"]


def __fold(time, flux, flux_ivar, period, duration, oversample):
    """
    Folds and bins the flux array at `period` and computes
    the maximum likelihood depth.

    Parameters
    ----------
    time : array-like
        Array with time measurements
    flux : array-like
        Array of measured fluxes at ``time``
    flux_ivar : array-like
        Inverse variance for each flux measurement
    period : scalar
        Period to each that light curve will be folded
    duration : scalar
        The transit duration to consider in the same units
        as ``time``
    oversample : int
        Resolution of the transit search

    Returns
    -------
    bins : array-like
        Bins over the phase
    depth : scalar
        Maximum likelihood depth
    depth_ivar : scalar
        Inverse variance at the maximum likelihood depth
    """

    d_bin = duration / oversample
    bins = np.arange(0, period+d_bin, d_bin)
    len_bins = len(bins)
    phase = time % period

    # Bin the folded data into a fine grid
    mean_flux_ivar, _ = histogram(phase, len_bins, range=(0, period),
                                  weights=flux_ivar)
    mean_flux, _ = histogram(phase, len_bins, range=(0, period),
                             weights=flux*flux_ivar)

    # Pre-compute some of the factors for the likelihood calculation
    sum_flux2_all = np.sum(flux**2 * flux_ivar)

    # Pad the arrays to deal with edge issues
    mean_flux = np.append(mean_flux, mean_flux[:oversample])
    mean_flux_ivar = np.append(mean_flux_ivar, mean_flux_ivar[:oversample])

    # Compute the maximum likelihood values and variance for in-transit (hin)
    # and out-of-transit (hout) flux estimates
    hin_ivar = np.cumsum(mean_flux_ivar)
    sum_ivar_all = hin_ivar[-oversample-1]
    hin_ivar = hin_ivar[oversample:] - hin_ivar[:-oversample]
    hin = np.cumsum(mean_flux)
    sum_flux_all = hin[-oversample-1]
    hin = hin[oversample:] - hin[:-oversample]

    hout_ivar = sum_ivar_all - hin_ivar
    hout = sum_flux_all - hin

    # Normalize in the in- and out-of-transit flux estimates
    hout /= hout_ivar
    hin /= hin_ivar

    # Compute the depth estimate
    depth = hout - hin
    depth_ivar = 1.0 / (1.0/hin_ivar + 1.0/hout_ivar)

    # Compute the log-likelihood
    hout2 = hout**2
    chi2 = sum_flux2_all - 2*hout*sum_flux_all + hout2*sum_ivar_all
    chi2 += ((hin**2-hout2) + 2*depth*hin)*hin_ivar

    return bins, depth, depth_ivar, -0.5*chi2


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

    if not HAS_FAST_HISTOGRAM:
        logging.warn("transit_periodogram has better performance with "
                     "the fast_histogram package installed")

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
    for duration in np.atleast_1d(np.abs(durations)):
        for i, period in enumerate(gen):
            depth, depth_var, ll, phase = _fold(time, flux, flux_ivar,
                                                sum_flux2_all, sum_flux_all,
                                                sum_ivar_all, period,
                                                duration, oversample,
                                                use_likelihood=use_likelihood)
            snr = depth / np.sqrt(depth_var)
            if use_likelihood:
                objective = ll
            else:
                objective = snr

            if objective > periodogram[i]:
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
