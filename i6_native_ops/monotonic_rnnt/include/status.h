#ifndef MONOTONIC_RNNT_STATUS_H
#define MONOTONIC_RNNT_STATUS_H

typedef enum {
    RNNT_STATUS_SUCCESS = 0,
    RNNT_STATUS_MEMOPS_FAILED = 1,
    RNNT_STATUS_INVALID_VALUE = 2,
    RNNT_STATUS_EXECUTION_FAILED = 3,
    RNNT_STATUS_UNKNOWN_ERROR = 4
} RNNTStatus;

/**
 * Returns a string containing a description of status that was passed in
 *  \param[in] status identifies which string should be returned
 *  \return C style string containing the text description
 **/
inline const char *rnntGetStatusString(RNNTStatus status) {
    switch (status) {
        case RNNT_STATUS_SUCCESS:
            return "no error";
        case RNNT_STATUS_MEMOPS_FAILED:
            return "cuda memcpy or memset failed";
        case RNNT_STATUS_INVALID_VALUE:
            return "invalid value";
        case RNNT_STATUS_EXECUTION_FAILED:
            return "execution failed";
        case RNNT_STATUS_UNKNOWN_ERROR:
        default:
            return "unknown error";
    }
}

#endif  // MONOTONIC_RNNT_STATUS_H
