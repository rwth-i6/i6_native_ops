#ifndef MONOTONIC_RNNT_ENTRYPOINT_H
#define MONOTONIC_RNNT_ENTRYPOINT_H

/** \file rnnt_entrypoint.h
 * Contains a simple C++ interface to call fast CPU and GPU based computation
 * of the RNNT loss.
 */

#include "options.h"
#include "status.h"
#include "workspace_manager.h"

extern "C" {

/**
 * \param [in]  workspace_manager Manager for handling memory access and structure.
 * \param [in]  options General options for the loss computation.
 * \param [out] costs 1-D array where the costs of each minibatch-sample are placed.
 * \param [out] gradients 1-D array where the gradients with respect to the activations
 *              are placed. May be nullptr.
 *
 *  \return Status information
 **/
RNNTStatus compute_rnnt_loss(RNNTWorkspaceManager &workspace_manager, RNNTOptions options, float *costs,
                             float *gradients);

}  // extern "C"

#endif  // MONOTONIC_RNNT_ENTRYPOINT_H
