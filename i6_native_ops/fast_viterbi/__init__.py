import os
import torch
from typing import Tuple
from pkg_resources import get_distribution

try:
    # Package is installed, so ops are already compiled
    __version__ = get_distribution('i6_native_ops').version
    import i6_native_ops.fast_viterbi.fast_viterbi_core as core
except Exception as e:
    # otherwise try to build locally
    from torch.utils.cpp_extension import load
    base_path = os.path.dirname(__file__)
    core = load(
        name="fast_viterbi_core",
        sources=[
            os.path.join(base_path, "binding.cpp"),
            os.path.join(base_path, "core.cu"),
        ],
        extra_include_paths=[os.path.join(base_path, "..", "common")],
	)

def align_viterbi(
    log_probs: torch.FloatTensor,
    fsa: Tuple[int, torch.IntTensor, torch.FloatTensor, torch.IntTensor],
    seq_lens: torch.IntTensor
) -> Tuple[torch.IntTensor, torch.FloatTensor]:
    """ Find best path with Viterbi algorithm.
    :param log_probs: log probabilities of emission model as a (B, T, F)
    :param fsa: weighted finite state automaton as a tuple consisting of:
        * number of states
        * a (4, E) tensor of integers specifying where each column consists of
            origin state, target state, emission idx and the index of the sequence
        * a (E,) tensor of floats holding the weight of each edge
        * a (2, B) tensor of starting and ending states for each automaton in the batch where
            the first row are starting states and the second the corresponding ending states
    :param seq_lens: (B,) tensor consisting of the sequence lengths
    :return: a sparse (B, T) tensor of the best sequences and a (B,) tensor of scores
    """
    log_probs = log_probs.transpose(0, 1).contiguous()
    num_states, edge_tensor, weight_tensor, start_end_states = fsa
    alignment, scores = core.fast_viterbi(
        log_probs, edge_tensor, weight_tensor,
        start_end_states, seq_lens, num_states
    )
    alignment_batch_major = alignment.transpose(0, 1).contiguous()
    return alignment_batch_major, scores
