import numpy as np
cimport numpy as np

cimport cython

from libc.math cimport sqrt
from libc.stdlib cimport malloc, free

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

IDTYPE = np.int32
ctypedef np.int32_t IDTYPE_t


cdef double compute_log_like(
        double flux_in,
        double flux_out,
        double ivar_in,
        double sum_flux2,
        double sum_flux,
        double sum_ivar,
):
    cdef double arg = flux_in - flux_out
    cdef double chi2 = sum_flux2 - 2*flux_out*sum_flux
    chi2 += flux_out*flux_out * sum_ivar
    chi2 -= arg*arg * ivar_in
    return -0.5*chi2


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void fold(
    # Inputs
    int N,              # Length of the time array
    double* time,       # The list of timestamps
    double* flux,       # The flux measured at ``time``
    double* ivar,       # The inverse variance of the flux array
    double sum_flux2,   # The precomputed value of sum(flux^2 * ivar)
    double sum_flux,    # The precomputed value of sum(flux * ivar)
    double sum_ivar,    # The precomputed value of sum(ivar)

    double period,      # The period to test in units of ``time``

    int K,              # Length of the durations array
    int* durations,     # The durations to test in units of ``d_bin``
    double d_bin,       # The width of the fine-grain bins to use in units of
                        # ``time``
    int oversample,     # The number of ``d_bin`` bins in the maximum duration

    int use_likelihood, # A flag indicating the periodogram type
                        # 0 - depth signal-to-noise
                        # 1 - log likelihood

    # Work arrays
    double* mean_flux,  # These are two work arrays that must be at least
    double* mean_ivar,  # n_bins = max(period/d_bin)+oversample long

    # Outputs
    double* best_objective,  # The value of the periodogram at maximum
    double* best_depth,      # The estimated depth at maximum
    double* best_depth_std,  # The uncertainty on ``best_depth``
    double* best_depth_snr,  # The signal-to-noise ratio of the depth estimate
    double* best_log_like,   # The log likelihood at maximum
    double* best_phase,      # The phase of the mid-transit time in units of
                             # ``time``
    double* best_duration    # The best fitting duration in units of ``time``
):

    cdef int n_bins = int(period / d_bin) + oversample + 1
    cdef int ind, n, k
    cdef double flux_in, flux_out, ivar_in, ivar_out, \
                depth, depth_std, depth_snr, log_like, objective

    for n in range(n_bins):
        mean_flux[n] = 0.0
        mean_ivar[n] = 0.0

    # Bin the data
    for n in range(N):
        ind = int((time[n] % period) / period * n_bins) + 1
        mean_flux[ind] += flux[n] * ivar[n]
        mean_ivar[ind] += ivar[n]

    # Pad the bins
    for n in range(1, oversample+1):
        ind = n_bins-oversample+n
        mean_flux[ind] = mean_flux[n]
        mean_ivar[ind] = mean_ivar[n]

    # Cumsum
    for n in range(1, n_bins):
        mean_flux[n] += mean_flux[n-1]
        mean_ivar[n] += mean_ivar[n-1]

    best_objective[0] = -np.inf
    for n in range(n_bins - oversample):
        for k in range(K):
            flux_in = mean_flux[n+durations[k]] - mean_flux[n]
            ivar_in = mean_ivar[n+durations[k]] - mean_ivar[n]
            flux_out = sum_flux - flux_in
            ivar_out = sum_ivar - ivar_in

            flux_in /= ivar_in
            flux_out /= ivar_out

            if use_likelihood:
                objective = compute_log_like(flux_in, flux_out, ivar_in,
                                             sum_flux2, sum_flux, sum_ivar)
            else:
                depth = flux_out - flux_in
                depth_std = sqrt(1.0 / ivar_in + 1.0 / ivar_out)
                objective = depth / depth_std

            if objective > best_objective[0]:
                if use_likelihood:
                    depth = flux_out - flux_in
                    depth_std = sqrt(1.0 / ivar_in + 1.0 / ivar_out)
                    depth_snr = depth / depth_std
                    log_like = objective
                else:
                    log_like = compute_log_like(flux_in, flux_out, ivar_in,
                                          sum_flux2, sum_flux,
                                          sum_ivar)
                    depth_snr = objective

                best_objective[0] = objective
                best_depth[0] = depth
                best_depth_std[0] = depth_std
                best_depth_snr[0] = depth_snr
                best_log_like[0] = log_like
                best_phase[0] = n * d_bin + 0.5 * durations[k] * d_bin
                best_duration[0] = durations[k] * d_bin


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def transit_periodogram_impl(
        np.ndarray[DTYPE_t, mode='c'] time_array,
        np.ndarray[DTYPE_t, mode='c'] flux_array,
        np.ndarray[DTYPE_t, mode='c'] flux_ivar_array,
        np.ndarray[DTYPE_t, mode='c'] period_array,
        np.ndarray[DTYPE_t, mode='c'] duration_array,
        int oversample,
        int use_likelihood=0):

    cdef double d_bin = max(duration_array) / oversample
    cdef double* periods = <double*>period_array.data
    cdef np.ndarray[IDTYPE_t] duration_int_array = \
            np.asarray(duration_array / d_bin, dtype=IDTYPE)
    cdef double* time = <double*>time_array.data
    cdef double* flux = <double*>flux_array.data
    cdef double* ivar = <double*>flux_ivar_array.data
    cdef int* durations = <int*>duration_int_array.data
    cdef int N = len(time_array)
    cdef int K = len(duration_array)

    cdef int p
    cdef int P = len(period_array)

    cdef double sum_flux2 = np.sum(flux_array * flux_array * flux_ivar_array)
    cdef double sum_flux = np.sum(flux_array * flux_ivar_array)
    cdef double sum_ivar = np.sum(flux_ivar_array)

    cdef np.ndarray[DTYPE_t] out_objective_array = np.empty(P, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t] out_depth_array     = np.empty(P, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t] out_depth_std_array = np.empty(P, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t] out_depth_snr_array = np.empty(P, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t] out_log_like_array  = np.empty(P, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t] out_phase_array     = np.empty(P, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t] out_duration_array  = np.empty(P, dtype=DTYPE)

    cdef double* out_objective = <double*>out_objective_array.data
    cdef double* out_depth     = <double*>out_depth_array.data
    cdef double* out_depth_std = <double*>out_depth_std_array.data
    cdef double* out_depth_snr = <double*>out_depth_snr_array.data
    cdef double* out_log_like  = <double*>out_log_like_array.data
    cdef double* out_phase     = <double*>out_phase_array.data
    cdef double* out_duration  = <double*>out_duration_array.data

    cdef int max_n_bins = int(np.max(period_array) / d_bin) + oversample + 1
    cdef double* mean_flux = <double*>malloc(max_n_bins*sizeof(double))
    if not mean_flux:
        raise MemoryError()
    cdef double* mean_ivar = <double*>malloc(max_n_bins*sizeof(double))
    if not mean_ivar:
        free(mean_flux)
        raise MemoryError()

    for p in range(P):
        fold(N, time, flux, ivar, sum_flux2, sum_flux,
             sum_ivar, periods[p], K, durations, d_bin,
             oversample, use_likelihood, mean_flux, mean_ivar,
             out_objective+p, out_depth+p, out_depth_std+p,
             out_depth_snr+p, out_log_like+p, out_phase+p, out_duration+p)

    free(mean_flux)
    free(mean_ivar)

    return (
        period_array, out_objective_array, out_log_like_array,
        out_depth_snr_array, out_depth_array, out_depth_std_array,
        out_phase_array, out_duration_array)
