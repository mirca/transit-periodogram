import numpy as np
cimport numpy as np

cimport cython
from libc.math cimport sqrt

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def fold(np.ndarray[DTYPE_t, mode='c'] time,
         np.ndarray[DTYPE_t, mode='c'] flux,
         np.ndarray[DTYPE_t, mode='c'] flux_ivar,
         double period,
         double duration,
         int oversample):

    cdef double d_bin = duration / oversample
    cdef int n_bins = int(period / d_bin) + oversample
    cdef np.ndarray[DTYPE_t] mean_flux = np.zeros(n_bins, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t] mean_ivar = np.zeros(n_bins, dtype=DTYPE)
    cdef int ind

    cdef double* time_d = <double*>time.data
    cdef double* flux_d = <double*>flux.data
    cdef double* ivar_d = <double*>flux_ivar.data
    cdef double* mean_flux_d = <double*>mean_flux.data
    cdef double* mean_ivar_d = <double*>mean_ivar.data

    cdef double sum_flux2_all = 0.0

    # Bin the data
    for n in range(len(time)):
        ind = int((time_d[n] % period) / period * n_bins);
        mean_flux_d[ind] += flux_d[n] * ivar_d[n]
        mean_ivar_d[ind] += ivar_d[n]
        sum_flux2_all += flux_d[n] * flux_d[n] * ivar_d[n]

    # Pad the bins
    for n in range(oversample):
        ind = n_bins-oversample+n
        mean_flux_d[ind] = mean_flux_d[n]
        mean_ivar_d[ind] = mean_ivar_d[n]

    # Cumsum
    for n in range(1, n_bins):
        mean_flux_d[n] += mean_flux_d[n-1]
        mean_ivar_d[n] += mean_ivar_d[n-1]

    # Compute the full sum
    cdef double sum_flux_all = mean_flux_d[n_bins-oversample-1]
    cdef double sum_ivar_all = mean_ivar_d[n_bins-oversample-1]

    cdef double hin
    cdef double hout
    cdef double hin_ivar
    cdef double hout_ivar
    cdef double depth
    cdef double depth_var
    cdef double depth_snr
    cdef double hout2
    cdef double best = -np.inf

    for n in range(n_bins - oversample):
        hin = mean_flux_d[n+oversample] - mean_flux_d[n]
        hin_ivar = mean_ivar_d[n+oversample] - mean_ivar_d[n]
        hout = sum_flux_all - hin
        hout_ivar = sum_ivar_all - hin_ivar

        hin /= hin_ivar
        hout /= hout_ivar

        depth = hout - hin
        depth_var = 1.0 / hin_ivar + 1.0 / hout_ivar
        depth_snr = depth / sqrt(depth_var)
        if depth_snr > best:
            hout2 = hout * hout
            chi2 = sum_flux2_all - 2*hout*sum_flux_all + hout2*sum_ivar_all
            chi2 += ((hin**2-hout2) + 2*depth*hin)*hin_ivar

            best_depth = depth
            best_depth_var = depth_var
            best_chi2 = chi2
            best_phase = (n + 0.5) * d_bin
