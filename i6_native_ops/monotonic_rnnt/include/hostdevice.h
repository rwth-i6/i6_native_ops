#ifndef MONOTONIC_RNNT_HOSTDEVICE_H
#define MONOTONIC_RNNT_HOSTDEVICE_H

#ifdef __CUDACC__
#define HOSTDEVICE __host__ __device__
#else
#define HOSTDEVICE
#endif

#endif  // MONOTONIC_RNNT_HOSTDEVICE_H
