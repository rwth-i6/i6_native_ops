import unittest
import torch

from i6_native_ops.monotonic_rnnt import monotonic_rnnt_loss


class TestCostAndGrad(unittest.TestCase):
    def test_cost_grad_values(self) -> None:
        acts = torch.tensor(
            [
                [0.6, 0.3, 0.1],
                [0.7, 0.1, 0.2],
                [0.5, 0.1, 0.4],
                [0.5, 0.4, 0.1],
                [0.5, 0.1, 0.4],
                [0.8, 0.1, 0.1],
                [0.4, 0.3, 0.3],
                [0.5, 0.1, 0.4],
                [0.7, 0.2, 0.1],
                [0.8, 0.1, 0.1],
                [0.3, 0.1, 0.6],
                [0.8, 0.1, 0.1],
            ],
            dtype=torch.float32,
        )
        acts = torch.log(acts)  # type: ignore

        labels = torch.tensor([[1, 2]], dtype=torch.int32)
        lengths = torch.tensor([4], dtype=torch.int32)
        label_lengths = torch.tensor([2], dtype=torch.int32)

        acts.requires_grad_(True)

        costs = monotonic_rnnt_loss(
            acts=acts,
            labels=labels,
            input_lengths=lengths,
            label_lengths=label_lengths,
            blank_label=0,
        )

        cost = costs.detach().numpy()[0]

        costs.backward()
        grads = acts.grad
        assert grads is not None

        expected_grads = torch.tensor(
            [
                [0.04, -0.14, 0.1],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.13, -0.19, 0.06],
                [-0.04, 0.04, -0.01],
                [0.0, 0.0, 0.0],
                [0.06, -0.1, 0.04],
                [0.01, 0.07, -0.08],
                [-0.06, 0.04, 0.02],
                [0.0, 0.0, 0.0],
                [0.14, 0.05, -0.19],
                [-0.11, 0.05, 0.05],
            ],
            dtype=torch.float32,
        )

        self.assertLess(abs(cost - 1.01), 1e-02)
        self.assertTrue(torch.allclose(grads, expected_grads, atol=1e-02))


class TestDatatypes(unittest.TestCase):
    def test_int64_labels_lengths(self) -> None:
        acts = torch.tensor(
            [
                [0.6, 0.3, 0.1],
                [0.7, 0.1, 0.2],
                [0.5, 0.1, 0.4],
                [0.5, 0.4, 0.1],
                [0.5, 0.1, 0.4],
                [0.8, 0.1, 0.1],
                [0.4, 0.3, 0.3],
                [0.5, 0.1, 0.4],
                [0.7, 0.2, 0.1],
                [0.8, 0.1, 0.1],
                [0.3, 0.1, 0.6],
                [0.8, 0.1, 0.1],
            ],
            dtype=torch.float32,
        )
        acts = torch.log(acts)  # type: ignore

        labels = torch.tensor([[1, 2]], dtype=torch.int64)
        lengths = torch.tensor([4], dtype=torch.int64)
        label_lengths = torch.tensor([2], dtype=torch.int64)

        acts.requires_grad_(True)

        costs = monotonic_rnnt_loss(
            acts=acts,
            labels=labels,
            input_lengths=lengths,
            label_lengths=label_lengths,
            blank_label=0,
        )

        cost = costs.detach().numpy()[0]
        self.assertLess(abs(cost - 1.01), 1e-02)


class TestAlignmentRestriction(unittest.TestCase):
    def test_alignment_restriction(self) -> None:
        acts = torch.tensor(
            [
                [0.6, 0.3, 0.1],
                [0.7, 0.1, 0.2],
                [0.5, 0.1, 0.4],
                [0.5, 0.4, 0.1],
                [0.5, 0.1, 0.4],
                [0.8, 0.1, 0.1],
                [0.4, 0.3, 0.3],
                [0.5, 0.1, 0.4],
                [0.7, 0.2, 0.1],
                [0.8, 0.1, 0.1],
                [0.3, 0.1, 0.6],
                [0.8, 0.1, 0.1],
            ],
            dtype=torch.float32,
        )
        acts = torch.log(acts)  # type: ignore

        labels = torch.tensor([[1, 2]], dtype=torch.int32)
        lengths = torch.tensor([4], dtype=torch.int32)
        label_lengths = torch.tensor([2], dtype=torch.int32)
        alignment = torch.tensor([[0, 1, 0, 2]], dtype=torch.int32)

        acts.requires_grad_(True)

        costs = monotonic_rnnt_loss(
            acts=acts,
            labels=labels,
            input_lengths=lengths,
            label_lengths=label_lengths,
            alignment=alignment,
            max_shift_from_alignment=1,
            blank_label=0,
        )

        cost = costs.detach().numpy()[0]

        self.assertLess(abs(cost - 1.22), 1e-02)

        alignment = torch.tensor([[1, 2, 0, 0]], dtype=torch.int32)

        acts.requires_grad_(True)

        costs = monotonic_rnnt_loss(
            acts=acts,
            labels=labels,
            input_lengths=lengths,
            label_lengths=label_lengths,
            alignment=alignment,
            max_shift_from_alignment=0,
            blank_label=0,
        )

        cost = costs.detach().numpy()[0]

        self.assertLess(abs(cost - 2.7), 1e-02)


if __name__ == "__main__":
    unittest.main()
