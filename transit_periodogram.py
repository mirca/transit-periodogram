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

    mean_flux_ivar, _ = np.histogram(phase, bins, weights=flux_ivar)
    mean_flux, _ = np.histogram(phase, bins, weights=flux*flux_ivar)
    mask_ivar = mean_flux_ivar > 0 # avoiding division by zero
    mean_flux[mask_ivar] /= mean_flux_ivar[mask_ivar]
    mean_flux = np.append(mean_flux, mean_flux[:oversample])
    mean_flux_ivar = np.append(mean_flux_ivar, mean_flux_ivar[:oversample])

    # computes the maximum likelihood heights
    # and their variance for in-transit (hin) and out-transit (hout)
    hin_ivar = np.cumsum(mean_flux_ivar)
    hin_ivar = hin_ivar[oversample:] - hin_ivar[:-oversample]
    hin = np.cumsum(mean_flux * mean_flux_ivar)
    hin = hin[oversample:] - hin[:-oversample]

    hout_ivar = np.sum(mean_flux_ivar) - hin_ivar
    hout = np.sum(mean_flux * mean_flux_ivar) - hin

    hout /= hout_ivar
    hin /= hin_ivar

    depth = hout - hin
    depth_ivar = 1.0 / (1.0/hin_ivar + 1.0/hout_ivar)

    return bins, depth, depth_ivar


def transit_periodogram(time, flux, periods, duration, flux_err=None,
                        oversample=10, progress=False):
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
    depth_snr = np.empty_like(periods)
    depths = np.empty_like(periods)
    depth_errs = np.empty_like(periods)
    phase = np.empty_like(periods)

    gen = periods
    if progress:
        gen = tqdm(periods, total=len(periods))

    for i, period in enumerate(gen):
        bins, depth, depth_ivar = _fold(time, flux, flux_ivar, period,
                                        duration, oversample)
        objective = depth * np.sqrt(depth_ivar) # depth S/N
        ind = np.argmax(objective)
        depth_snr[i] = objective[ind]
        depths[i] = depth[ind]
        depth_errs[i] = np.sqrt(1./depth_ivar[ind])
        phase[i] = bins[ind] + 0.5 * duration

    return periods, depth_snr, depths, depth_errs, phase
