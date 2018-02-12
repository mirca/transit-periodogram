# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["transit_periodogram"]

import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, *args, **kwargs: x


def fold(time, flux, flux_ivar, period, duration, oversample):
    d_bin = 0.5 * duration / oversample
    bins = np.arange(0, period+d_bin, d_bin)
    phase = time % period
    mean_flux_ivar, _ = np.histogram(phase, bins, weights=flux_ivar)
    mean_flux, _ = np.histogram(phase, bins, weights=flux*flux_ivar)
    mask_ivar = mean_flux_ivar > 0
    mean_flux[mask_ivar] /= mean_flux_ivar[mask_ivar]

    mean_flux = np.append(mean_flux, mean_flux[:oversample])
    mean_flux_ivar = np.append(mean_flux_ivar, mean_flux_ivar[:oversample])

    hin_ivar = np.cumsum(mean_flux_ivar)
    hin_ivar = hin_ivar[oversample:] - hin_ivar[:-oversample]
    hin = np.cumsum(mean_flux * mean_flux_ivar)
    hin = hin[oversample:] - hin[:-oversample]

    hout_ivar = np.sum(mean_flux_ivar) - hin_ivar
    hout = np.sum(mean_flux * mean_flux_ivar) - hin
    hout /= hout_ivar
    hin /= hin_ivar

    depth = hout - hin
    depth_ivar = 1.0/(1.0 / hin_ivar + 1.0 / hout_ivar)

    return bins, depth, depth_ivar


def transit_periodogram(time, flux, periods, duration, flux_err=None,
                        oversample=10, progress=False):
    """Compute the transit periodogram for some dataz

    TODO: better docs

    Args:
        time: An array of times
        flux: The fluxes measured at ``time``
        periods: The periods to search in the same units as ``time``
        duration: The transit duration to consider in the same units
            as ``time``
        flux_err: The uncertainties on ``flux``

    Returns:
        osiadfj

    """
    if flux_err is None:
        flux_err = np.ones_like(flux)

    time = np.atleast_1d(time)
    flux = np.atleast_1d(flux)
    flux_err = np.atleast_1d(flux_err)
    flux_ivar = 1.0 / flux_err**2

    periods = np.atleast_1d(periods)
    periodogram = np.empty_like(periods)
    depths = np.empty_like(periods)
    depth_ivars = np.empty_like(periods)
    phase = np.empty_like(periods)

    gen = periods
    if progress:
        gen = tqdm(periods, total=len(periods))

    for i, period in enumerate(gen):
        bins, depth, depth_ivar = fold(time, flux, flux_ivar, period,
                                       duration, oversample)
        objective = depth * np.sqrt(depth_ivar)
        ind = np.argmax(objective)

        periodogram[i] = objective[ind]
        depths[i] = depth[ind]
        depth_ivars[i] = depth_ivar[ind]
        phase[i] = bins[ind] + 0.5*duration

    return periods, periodogram, depths, depth_ivars, phase
