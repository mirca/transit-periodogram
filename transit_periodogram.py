# -*- coding: utf-8 -*-

from __future__ import division, print_function

import numpy as np
import six

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, *args, **kwargs: x


__all__ = ["transit_periodogram"]


def _fold(time, flux, flux_ivar, period, duration, oversample):
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

    if not isinstance(oversample, six.integer_types):
        raise ValueError("oversample must be an integer, got {0}"
                         .format(oversample))

    d_bin = duration / oversample
    bins = np.arange(0, period+d_bin, d_bin)
    phase = time % period

    # Bin the folded data into a fine grid
    mean_flux_ivar, _ = np.histogram(phase, bins, weights=flux_ivar)
    mean_flux, _ = np.histogram(phase, bins, weights=flux*flux_ivar)

    # Pre-compute some of the factors for the likelihood calculation
    sum_ivar_all = np.sum(mean_flux_ivar)
    sum_flux_all = np.sum(mean_flux)
    sum_flux2_all = np.sum(flux**2 * flux_ivar)

    # Pad the arrays to deal with edge issues
    mean_flux = np.append(mean_flux, mean_flux[:oversample])
    mean_flux_ivar = np.append(mean_flux_ivar, mean_flux_ivar[:oversample])

    # Compute the maximum likelihood values and variance for in-transit (hin)
    # and out-of-transit (hout) flux estimates
    hin_ivar = np.cumsum(mean_flux_ivar)
    hin_ivar = hin_ivar[oversample:] - hin_ivar[:-oversample]
    hin = np.cumsum(mean_flux)
    hin = hin[oversample:] - hin[:-oversample]

    # Compute the in transit sums used to compute the likelihood
    sum_ivar_in = np.array(hin_ivar)
    sum_flux_in = np.array(hin)

    # Compute the out of transit flux estimate
    hout_ivar = np.sum(mean_flux_ivar) - hin_ivar
    hout = np.sum(mean_flux) - hin

    # Normalize in the in- and out-of-transit flux estimates
    hout /= hout_ivar
    hin /= hin_ivar

    # Compute the depth estimate
    depth = hout - hin
    depth_ivar = 1.0 / (1.0/hin_ivar + 1.0/hout_ivar)

    # Compute the log-likelihood
    hout2 = hout**2
    chi2 = sum_flux2_all - 2*hout*sum_flux_all + hout2*sum_ivar_all
    chi2 += (hin**2-hout2)*sum_ivar_in + 2*depth*sum_flux_in

    return bins, depth, depth_ivar, -0.5*chi2


def transit_periodogram(time, flux, periods, duration, flux_err=None,
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
    duration : scalar
        The transit duration to consider in the same units
        as ``time``
    flux_err : array-like
        The uncertainties on ``flux``

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
    if method is None:
        method = "snr"
    allowed_methods = ["snr", "likelihood"]
    if method not in allowed_methods:
        raise ValueError("unrecognized method '{0}'\nallowed methods are: {1}"
                         .format(method, allowed_methods))

    time = np.atleast_1d(time)
    flux = np.atleast_1d(flux)

    if time.shape != flux.shape:
        raise ValueError("time and flux arrays have different shapes: "
                         "{0}, {1}".format(time.shape, flux.shape))
    if flux_err is None:
        flux_err = np.ones_like(flux)

    flux_err = np.atleast_1d(flux_err)
    flux_ivar = 1.0 / flux_err**2

    periods = np.atleast_1d(periods)
    periodogram = np.empty_like(periods)
    log_likelihood = np.empty_like(periods)
    depth_snr = np.empty_like(periods)
    depths = np.empty_like(periods)
    depth_errs = np.empty_like(periods)
    phase = np.empty_like(periods)

    gen = periods
    if progress:
        gen = tqdm(periods, total=len(periods))

    for i, period in enumerate(gen):
        bins, depth, depth_ivar, ll = _fold(time, flux, flux_ivar, period,
                                            duration, oversample)
        snr = depth * np.sqrt(depth_ivar)
        if method == "snr":
            objective = snr
        else:
            objective = ll

        ind = np.argmax(objective)
        periodogram[i] = objective[ind]
        log_likelihood[i] = ll[ind]
        depth_snr[i] = snr[ind]
        depths[i] = depth[ind]
        depth_errs[i] = np.sqrt(1./depth_ivar[ind])
        phase[i] = bins[ind] + 0.5 * duration

    return (
        periods, periodogram, log_likelihood, depth_snr, depths, depth_errs,
        phase
    )
