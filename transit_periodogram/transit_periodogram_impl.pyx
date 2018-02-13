import numpy as np
cimport numpy as np

cimport cython
from libc.math cimport sqrt

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

IDTYPE = np.int32
ctypedef np.int32_t IDTYPE_t

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def fold(np.ndarray[DTYPE_t, mode='c'] time,
         np.ndarray[DTYPE_t, mode='c'] flux,
         np.ndarray[DTYPE_t, mode='c'] flux_ivar,
         double sum_flux2_all,
         double sum_flux_all,
         double sum_ivar_all,
         double period,
         np.ndarray[DTYPE_t, mode='c'] durations,
         int oversample,
         int use_likelihood=0):

    cdef double d_bin = max(durations) / oversample
    cdef np.ndarray[IDTYPE_t] durations_dbin = np.asarray(durations / d_bin,
                                                          dtype=IDTYPE)
    cdef int n_bins = int(period / d_bin) + oversample
    cdef np.ndarray[DTYPE_t] mean_flux = np.zeros(n_bins, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t] mean_ivar = np.zeros(n_bins, dtype=DTYPE)

    cdef double* time_d = <double*>time.data
    cdef double* flux_d = <double*>flux.data
    cdef double* ivar_d = <double*>flux_ivar.data
    cdef double* mean_flux_d = <double*>mean_flux.data
    cdef double* mean_ivar_d = <double*>mean_ivar.data
    cdef int* durations_dbin_d = <int*>durations_dbin.data

    cdef int ind
    cdef int n
    cdef int k
    cdef int N = len(time)
    cdef int K = len(durations)

    # Bin the data
    for n in range(N):
        ind = int((time_d[n] % period) / period * n_bins);
        mean_flux_d[ind] += flux_d[n] * ivar_d[n]
        mean_ivar_d[ind] += ivar_d[n]

    # Pad the bins
    for n in range(oversample):
        ind = n_bins-oversample+n
        mean_flux_d[ind] = mean_flux_d[n]
        mean_ivar_d[ind] = mean_ivar_d[n]

    # Cumsum
    for n in range(1, n_bins):
        mean_flux_d[n] += mean_flux_d[n-1]
        mean_ivar_d[n] += mean_ivar_d[n-1]

    cdef double ll
    cdef double hin
    cdef double hout
    cdef double hin_ivar
    cdef double hout_ivar
    cdef double depth
    cdef double depth_var
    cdef double depth_snr
    cdef double hout2
    cdef double best = -np.inf
    cdef double obj

    cdef double best_depth, best_depth_var, best_ll, best_phase, best_duration

    for n in range(n_bins - oversample):
        for k in range(K):
            hin = mean_flux_d[n+durations_dbin_d[k]] - mean_flux_d[n]
            hin_ivar = mean_ivar_d[n+durations_dbin_d[k]] - mean_ivar_d[n]
            hout = sum_flux_all - hin
            hout_ivar = sum_ivar_all - hin_ivar

            hin /= hin_ivar
            hout /= hout_ivar

            depth = hout - hin
            if use_likelihood:
                hout2 = hout * hout
                obj = sum_flux2_all - 2*hout*sum_flux_all + hout2*sum_ivar_all
                obj += ((hin**2-hout2) + 2*depth*hin)*hin_ivar
                obj *= -0.5
            else:
                depth_var = 1.0 / hin_ivar + 1.0 / hout_ivar
                obj = depth / sqrt(depth_var)

            if obj > best:
                if use_likelihood:
                    depth_var = 1.0 / hin_ivar + 1.0 / hout_ivar
                    depth_snr = depth / sqrt(depth_var)
                    ll = obj
                else:
                    hout2 = hout * hout
                    ll = sum_flux2_all - 2*hout*sum_flux_all + hout2*sum_ivar_all
                    ll += ((hin**2-hout2) + 2*depth*hin)*hin_ivar
                    ll *= -0.5

                best = obj
                best_depth = depth
                best_depth_var = depth_var
                best_ll = ll
                best_phase = n * d_bin
                best_duration = durations_dbin_d[k] * d_bin

    return best_depth, best_depth_var, best_ll, best_phase, best_duration


#@cython.cdivision(True)
#@cython.boundscheck(False)
#@cython.wraparound(False)
#def transit_periodogram(np.ndarray[DTYPE_t, mode='c'] time,
#                        np.ndarray[DTYPE_t, mode='c'] flux,
#                        np.ndarray[DTYPE_t, mode='c'] periods,
#                        np.ndarray[DTYPE_t, mode='c'] durations,
#                        np.ndarray[DTYPE_t, mode='c'] flux_ivar,
#                        int oversample,
#                        int use_likelihood=0):
#
#    cdef double* time_d = <double*>time.data
#    cdef double* flux_d = <double*>flux.data
#    cdef double* periods_d = <double*>periods.data
#    cdef double* durations_d = <double*>durations.data
#
