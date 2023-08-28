import torch

import numpy as np
from torch.nn import Module

from .hier import Hierarchy
from .metrics import LCAMetric


class MarginLoss(Module):
    """Computes soft or hard margin loss for given a margin function."""

    def __init__(
            self, tree: Hierarchy,
            with_leaf_targets: bool,
            hardness: str = 'soft',
            margin: str = 'depth_dist',
            tau: float = 1.0):
        super().__init__()
        if hardness not in ('soft', 'hard'):
            raise ValueError('unknown hardness', hardness)
        n = tree.num_nodes()
        label_order = tree.leaf_subset() if with_leaf_targets else np.arange(n)

        # Construct array label_margin[gt_label, pr_node].
        if margin in ('edge_dist', 'depth_dist'):
            # label_margin = metrics.edge_dist(tree, label_order[:, None], np.arange(n)[None, :])
            depth = tree.depths()
            margin_arr = LCAMetric(tree, depth).dist(label_order[:, None], np.arange(n))
        elif margin == 'incorrect':
            is_ancestor = tree.ancestor_mask()
            is_correct = is_ancestor[:, label_order].T
            margin_arr = 1 - is_correct
        elif margin == 'info_dist':
            # TODO: Does natural log make most sense here?
            info = np.log(tree.num_leaf_nodes() / tree.num_leaf_descendants())
            margin_arr = LCAMetric(tree, info).dist(label_order[:, None], np.arange(n))
        elif margin == 'depth_deficient':
            depth = tree.depths()
            margin_arr = LCAMetric(tree, depth).deficient(label_order[:, None], np.arange(n))
        elif margin == 'log_depth_f1_error':
            depth = tree.depths()
            margin_arr = np.log(1 - LCAMetric(tree, depth).f1(label_order[:, None], np.arange(n)))
        else:
            raise ValueError('unknown margin', margin)

        # correct_margins = margin_arr[np.arange(len(label_order)), label_order]
        # if not np.all(correct_margins == 0):
        #     raise ValueError('margin with self is not zero', correct_margins)

        self.hardness = hardness
        self.tau = tau
        self.label_order = torch.from_numpy(label_order)
        self.margin = torch.from_numpy(margin_arr)

    def _apply(self, fn):
        super()._apply(fn)
        self.label_order = fn(self.label_order)
        self.margin = fn(self.margin)
        return self

    def forward(self, scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        label_node = labels if self.label_order is None else self.label_order[labels]
        label_score = scores.gather(-1, label_node.unsqueeze(-1)).squeeze(-1)
        label_margin = self.margin[labels, :]
        if self.hardness == 'soft':
            loss = -label_score + torch.logsumexp(scores + self.tau * label_margin, axis=-1)
        elif self.hardness == 'hard':
            # loss = -label_score + torch.max(torch.relu(scores + self.tau * label_margin), axis=-1)[0]
            loss = torch.relu(torch.max(scores - label_score.unsqueeze(-1) + self.tau * label_margin, axis=-1)[0])
        else:
            assert False
        return torch.mean(loss)
