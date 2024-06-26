import os
import torch  # needed to find pytorch specific libs
from pkg_resources import get_distribution
from typing import Tuple

try:
    # Package is installed, so ops are already compiled
    __version__ = get_distribution('i6_native_ops').version
    from . import fbw_core as core
except Exception as e:
    # otherwise try to build locally
    from torch.utils.cpp_extension import load
    base_path = os.path.dirname(__file__)
    core = load(
        name="fbw_core",
        sources=[
            os.path.join(base_path, "fbw_torch.cpp"),
            os.path.join(base_path, "fbw_op.cu"),
        ],
        extra_include_paths=[
            base_path,
            os.path.join(base_path, "..", "common")
        ],
    )


class FastBaumWelchLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, am_scores, fsa, seq_lens):
        num_states, edge_tensor, weight_tensor, start_end_states = fsa

        grad, loss = core.fbw(
            am_scores, edge_tensor, weight_tensor,
            start_end_states, seq_lens, int(num_states),
            core.DebugOptions()
        )
        ctx.save_for_backward(grad)
        return loss
    
    @staticmethod
    def backward(ctx, grad_loss):
        # negative log prob -> prob
        grad = ctx.saved_tensors[0].neg().exp()
        return grad, None, None


def fbw_loss(
    log_probs: torch.FloatTensor,
    fsa: Tuple[int, torch.IntTensor, torch.FloatTensor, torch.IntTensor],
    seq_lens: torch.IntTensor
) -> torch.FloatTensor:
    """ 
    Computes negative log likelihood of an emission model given an HMM finite state automaton.
    The corresponding gradient with respect to the emission model is automatically backpropagated.
    :param log_probs: log probabilities of emission model as a [B, T, F] tensor
    :param fsa: weighted finite state automaton as a tuple consisting of:
        * number of states
        * a (4, E) tensor of integers specifying where each column consists of
            origin state, target state, emission idx and the index of the sequence
        * a (E,) tensor of floats holding the weight of each edge
        * a (2, B) tensor of starting and ending states for each automaton in the batch where
            the first row are starting states and the second the corresponding ending states
    :param seq_lens: (B,) tensor consisting of the sequence lengths
    :return: (B,) tensor of loss values
    """
    neg_log_probs = log_probs.neg().transpose(0, 1).contiguous() # [T, B, F]
    loss = FastBaumWelchLoss.apply(neg_log_probs, fsa, seq_lens)
    return loss