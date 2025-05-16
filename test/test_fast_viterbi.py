import unittest
import torch

from i6_native_ops.fast_viterbi import align_viterbi

class TestFastViterbi(unittest.TestCase):

    def test_best_sequence(self):
        log_probs = (
            torch.tensor([
                [0.9, 0.1],
                [0.9, 0.1],
                [0.4, 0.6],
                [0.1, 0.9],
                [0.1, 0.9],
            ], device="cuda", dtype=torch.float32)
            .unsqueeze(1)
            .log()
        )
        edges = (
            torch.tensor([
                # from, to, emission_idx, sequence_idx
                [0, 0, 0, 0], # loop from 0 to 0, emit label 0
                [0, 1, 0, 0], # forward from 0 to 1, emit 0
                [1, 1, 1, 0]], # loop from 1 to 1, emit 1
                device="cuda", dtype=torch.int32)
            .transpose(0, 1).contiguous()
        )
        weights = torch.tensor([1, 1, 1], device="cuda", dtype=torch.float32)
        start_end_states = torch.tensor([[0], [1]], dtype=torch.int32, device="cuda")
        seq_lens = torch.tensor([5], dtype=torch.int32, device="cuda")

        fsa = (2, edges, weights, start_end_states)

        output, scores = align_viterbi(log_probs, fsa, seq_lens)
        best_sequence = list(output[:,0])
        score = float(scores[0])

        self.assertEqual(best_sequence, [0, 0, 1, 1, 1])

if __name__ == "__main__":
    unittest.main()