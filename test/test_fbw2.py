import itertools
import unittest
import torch

from i6_native_ops.fbw2 import fbw2_loss


def fbw_loss_auto(log_probs, fsa, seq_lens) -> torch.Tensor:
    """
    Manual computation of the sequence probability.
    Currently only supports one sequence at a time.
    """
    num_seqs = seq_lens.shape[0]
    assert num_seqs == 1
    num_frames = seq_lens[0]
    num_states, num_edges, edges, weights, start_end_states = fsa
    num_states = num_states[0]
    start_state, end_state = start_end_states.tolist()
    edges = edges.transpose(0, 1).to(device="cpu")
    loss = torch.tensor([0.0], dtype=torch.float32, device="cuda")
    probs = log_probs.exp()
    edge_to_eidx_mapping = {
        (int(from_state), int(to_state)): idx for from_state, to_state, idx, _ in edges
    }
    edge_to_weight_mapping = {
        (int(from_state), int(to_state)): weight
        for (from_state, to_state, *_), weight in zip(edges, weights)
    }

    # iterate over all possible state sequences
    for inner_state_seq in itertools.product(range(num_states), repeat=num_frames - 1):
        state_seq = (start_state[0],) + inner_state_seq + (end_state[0],)
        sequence_prob = torch.tensor([1.0], dtype=torch.float32, device="cuda")
        for t, edge in enumerate(itertools.pairwise(state_seq)):
            eidx = edge_to_eidx_mapping.get(edge, None)
            if eidx is None:
                # edge between this pair of states
                sequence_prob *= 0.0
                break
            weight = edge_to_weight_mapping.get(edge)
            emission_prob = probs[0, t, eidx]
            sequence_prob *= weight.neg().exp() * emission_prob
        loss += sequence_prob
    return loss.log().neg()


class TestFastBaumWelch(unittest.TestCase):
    def setUp(self):
        self.log_probs = (
            torch.tensor(
                [
                    [0.9, 0.1],
                    [0.9, 0.1],
                    [0.4, 0.6],
                    [0.1, 0.9],
                    [0.1, 0.9],
                ],
                device="cuda",
                dtype=torch.float32,
                requires_grad=True,
            )
            .unsqueeze(0)
            .log()
        )
        self.log_probs_copy = self.log_probs.clone().detach().requires_grad_(True)
        edges = (
            torch.tensor(
                [
                    # from, to, emission_idx, sequence_idx
                    [0, 0, 0, 0],  # loop from 0 to 0, emit label 0
                    [0, 1, 0, 0],  # forward from 0 to 1, emit 0
                    [1, 1, 1, 0],  # loop from 1 to 1, emit 1
                ],
                device="cuda",
                dtype=torch.int32,
            )
            .transpose(0, 1)
            .contiguous()
        )
        weights = torch.tensor([1, 1, 1], device="cuda", dtype=torch.float32)
        start_end_states = torch.tensor([[0], [1]], dtype=torch.int32, device="cuda")
        self.seq_lens = torch.tensor([5], dtype=torch.int32, device="cuda")

        self.fsa = (
            torch.tensor([2], dtype=torch.int32),
            torch.tensor([3], dtype=torch.int32),
            edges,
            weights,
            start_end_states,
        )

        self.log_probs.retain_grad()

    def test_grad(self):
        """Test whether the CUDA loss agrees with the manual one."""
        # CUDA loss
        loss = fbw2_loss(self.log_probs, self.fsa, self.seq_lens)
        loss.sum().backward()
        grad = self.log_probs.grad

        # automatic loss
        loss_auto = fbw_loss_auto(self.log_probs_copy, self.fsa, self.seq_lens)
        loss_auto.sum().backward()
        grad_auto = self.log_probs_copy.grad

        self.assertTrue(torch.isclose(loss, loss_auto).all())
        self.assertTrue(torch.isclose(grad, grad_auto).all())

    def test_memory(self):
        """Test for memory leaks in the CUDA loss."""
        free_mem_pre_op_call = torch.cuda.mem_get_info()[0]
        fbw2_loss(self.log_probs, self.fsa, self.seq_lens)
        free_mem_post_op_call = torch.cuda.mem_get_info()[0]

        self.assertEqual(
            free_mem_pre_op_call,
            free_mem_post_op_call,
            msg="Memory leak detected in CUDA FBW op",
        )


if __name__ == "__main__":
    unittest.main()
