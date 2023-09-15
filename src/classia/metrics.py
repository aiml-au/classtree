from typing import Dict, List, Tuple
import numpy as np

from .hier import Hierarchy, FindLCA, uniform_leaf


class LCAMetric:

    def __init__(self, tree: Hierarchy, value: np.ndarray):
        self.value = value
        self.find_lca = FindLCA(tree)

    def value_at_lca(self, gt: np.ndarray, pr: np.ndarray) -> np.ndarray:
        lca = self.find_lca(gt, pr)
        return self.value[lca]

    def value_at_gt(self, gt: np.ndarray, pr: np.ndarray) -> np.ndarray:
        # TODO: Avoid broadcasting of unused array?
        gt, _ = np.broadcast_arrays(gt, pr)
        return self.value[gt]

    def value_at_pr(self, gt: np.ndarray, pr: np.ndarray) -> np.ndarray:
        # TODO: Avoid broadcasting of unused array?
        _, pr = np.broadcast_arrays(gt, pr)
        return self.value[pr]

    def deficient(self, gt: np.ndarray, pr: np.ndarray) -> np.ndarray:
        lca = self.find_lca(gt, pr)
        return self.value[gt] - self.value[lca]

    def excess(self, gt: np.ndarray, pr: np.ndarray) -> np.ndarray:
        lca = self.find_lca(gt, pr)
        return self.value[pr] - self.value[lca]

    def dist(self, gt: np.ndarray, pr: np.ndarray) -> np.ndarray:
        lca = self.find_lca(gt, pr)
        excess = self.value[pr] - self.value[lca]
        deficient = self.value[gt] - self.value[lca]
        return excess + deficient

    def recall(self, gt: np.ndarray, pr: np.ndarray) -> np.ndarray:
        lca = self.find_lca(gt, pr)
        gt_value = self.value[gt]
        lca_value = self.value[lca]
        with np.errstate(invalid='ignore'):
            return np.where((lca_value == 0) & (gt_value == 0), 1.0, lca_value / gt_value)

    def precision(self, gt: np.ndarray, pr: np.ndarray) -> np.ndarray:
        lca = self.find_lca(gt, pr)
        pr_value = self.value[pr]
        lca_value = self.value[lca]
        with np.errstate(invalid='ignore'):
            return np.where((lca_value == 0) & (pr_value == 0), 1.0, lca_value / pr_value)

    def f1(self, gt: np.ndarray, pr: np.ndarray) -> np.ndarray:
        lca = self.find_lca(gt, pr)
        gt_value = self.value[gt]
        pr_value = self.value[pr]
        lca_value = self.value[lca]
        with np.errstate(invalid='ignore'):
            r = np.where((lca_value == 0) & (gt_value == 0), 1.0, lca_value / gt_value)
            p = np.where((lca_value == 0) & (pr_value == 0), 1.0, lca_value / pr_value)
        with np.errstate(divide='ignore'):
            return 2 / (1 / r + 1 / p)


def UniformLeafInfoMetric(tree: Hierarchy) -> LCAMetric:
    info = -np.log2(uniform_leaf(tree))
    return LCAMetric(tree, info)


def DepthMetric(tree: Hierarchy) -> LCAMetric:
    return LCAMetric(tree, tree.depths())


class IsCorrect:

    def __init__(self, tree: Hierarchy):
        self.find_lca = FindLCA(tree)
        # self.depths = tree.depths()

    def __call__(self, gt: np.ndarray, pr: np.ndarray) -> np.ndarray:
        lca = self.find_lca(gt, pr)
        # depth_gt = self.depths[gt]
        # depth_pr = self.depths[pr]
        # depth_lca = self.depths[lca]
        # # Correct if gt is below pr or pr is below gt.
        # # If this is the case, lca == gt or lca == pr.
        # return (depth_lca == depth_gt) | (depth_lca == depth_pr)
        return (lca == gt) | (lca == pr)


def operating_curve(
        example_scores: List[np.ndarray],
        example_metrics: Dict[str, List[np.ndarray]],
        ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Obtains operating curve for set of metrics.

    For each field in example_metrics, `example_scores[i]` and `example_metrics[field][i]`
    are arrays of the same length, ordered descending by score.
    This can be obtained using `infer.prediction_sequence()`.
    """
    # Assert that scores are sorted (descending) per example.
    for seq in example_scores:
        if not np.all(np.diff(seq) <= 0):
            raise ValueError('scores must be strictly descending', seq)

    # # Check that all scores have identical start.
    # init_scores = np.array([seq[0] for seq in example_scores])
    # unique_init_scores = np.unique(init_scores)
    # if not len(unique_init_scores) == 1:
    #     raise ValueError('initial scores are not equal', unique_init_scores)
    # init_score, = unique_init_scores

    # Obtain order of scores.
    # Note: Could do a merge sort here, since each array is already sorted.
    step_scores = np.concatenate([seq[1:] for seq in example_scores])
    step_order = np.argsort(-step_scores)
    step_scores = step_scores[step_order]
    # Identify first element in each group of scores.
    n = len(step_scores)
    _, first_index = np.unique(-step_scores, return_index=True)
    group_scores = step_scores[first_index]
    # group_scores = np.concatenate(([init_score], step_scores[first_index]))
    last_index = np.concatenate((first_index[1:], [n])) - 1
    group_totals = {}
    for field, example_values in example_metrics.items():
        # Convert to float since np.diff() treats bools as mod 2 arithmetic.
        example_values = [seq.astype(float) for seq in example_values]
        total_init = np.sum([seq[0] for seq in example_values])
        total_deltas = np.concatenate([np.diff(seq) for seq in example_values])[step_order]
        group_totals[field] = np.concatenate(([total_init], total_init + np.cumsum(total_deltas)[last_index]))
    return group_scores, group_totals