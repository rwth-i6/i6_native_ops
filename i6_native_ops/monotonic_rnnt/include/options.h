#ifndef MONOTONIC_RNNT_OPTIONS_H
#define MONOTONIC_RNNT_OPTIONS_H

// forward declare of CUDA typedef to avoid needing to pull in CUDA headers
typedef struct CUstream_st *CUstream;

typedef enum { RNNT_CPU = 0, RNNT_GPU = 1 } rnntComputeLocation;

/**
 * Structure containing general computation options.
 **/
struct RNNTOptions {
    // The maximum number of threads that can be used
    int num_threads;

    // used when computing on GPU, which stream the kernels should be launched in
    CUstream stream;

    // the label value/index that the RNNT calculation should use as the blank label
    int blank_label;

    /// indicates where the rnnt calculation should take place {RNNT_CPU | RNNT_GPU}
    rnntComputeLocation loc;
};

#endif  // MONOTONIC_RNNT_OPTIONS_H
