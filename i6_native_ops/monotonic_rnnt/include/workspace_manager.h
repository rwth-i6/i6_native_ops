#ifndef MONOTONIC_RNNT_WORKSPACE_MANAGER_H
#define MONOTONIC_RNNT_WORKSPACE_MANAGER_H

class RNNTWorkspaceManager {
   public:
    RNNTWorkspaceManager() = default;

    RNNTWorkspaceManager(const RNNTWorkspaceManager &) = delete;

    virtual ~RNNTWorkspaceManager() = default;
};

#endif  // MONOTONIC_RNNT_WORKSPACE_MANAGER_H
