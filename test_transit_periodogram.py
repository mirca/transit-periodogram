import numpy as np
import pytest
from transit_periodogram import transit_periodogram


@pytest.mark.parametrize("method", [("snr"), ("likelihood")])
def test_transit_periodogram(method):
    np.random.seed(0)
    time = np.linspace(0, 60, 3000)
    period = 5
    transit_depth = 0.01
    transit_time = 2.5
    transit_duration = 0.125

    flux_err = 0.01 + np.zeros_like(time)
    flux = np.ones_like(time)
    flux[np.abs((time - transit_time + 0.5*period)
         % period - 0.5*period) < 0.5*transit_duration] = 1.0 - transit_depth
    flux += flux_err * np.random.randn(len(flux))

    df = 0.5 / (time.max() - time.min())
    freq = np.arange(1./8, 1./4, 0.02*df)
    periods = 1.0 / freq

    periods, objective, log_likelihood, depth_snr, depths, depth_errs, phase,\
            best_durations = transit_periodogram(time, flux, periods,
                                                 transit_duration, flux_err,
                                                 method=method)

    ind = np.argmax(depth_snr)
    assert abs(1/periods[ind] - 1/period) < 0.02 * df
    assert abs(depths[ind] - transit_depth)/depth_errs[ind] < 1
