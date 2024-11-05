import unittest
import torch

from i6_native_ops.fbw import fbw_loss


class TestFastBaumWelch(unittest.TestCase):
    def test_grad(self):
        log_probs = (
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
        seq_lens = torch.tensor([5], dtype=torch.int32, device="cuda")

        fsa = (2, edges, weights, start_end_states)

        log_probs.retain_grad()

        free_mem_pre_op_call = torch.cuda.mem_get_info()[0]
        loss = fbw_loss(log_probs, fsa, seq_lens)
        free_mem_post_op_call = torch.cuda.mem_get_info()[0]

        self.assertEqual(
            free_mem_pre_op_call,
            free_mem_post_op_call,
            msg="Memory leak detected in CUDA FBW op",
        )

        loss.sum().backward()
        grad = log_probs.grad

        self.assertTrue(
            torch.isclose(
                grad.sum(-1).neg(), torch.full(log_probs.shape[:2], 1.0, device="cuda")
            ).all()
        )


if __name__ == "__main__":
    unittest.main()
