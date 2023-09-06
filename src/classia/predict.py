from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch.nn.functional import softmax
from torchtext.models import ROBERTA_BASE_ENCODER
from torchvision.transforms.v2 import CenterCrop, ToImageTensor, ConvertImageDtype, Normalize, Compose

from .hier import make_hierarchy_from_edges, SumDescendants
from .models import ClassiaImageModelV1, ClassiaTextModelV1


MIN_THRESHOLD = 0.3

def predict_images(files, *, models_dir, model_name, batch_size=8, device='cuda'):
    with open(f"{models_dir}/{model_name}/labels.txt", "r") as f:
        lines = [line.strip() for line in f.readlines()]
        labels = lines
    with open(f"{models_dir}/{model_name}/tree.csv", "r") as f:
        lines = [line.strip() for line in f.readlines()]
        edges = [line.split(",") for line in lines]
        tree, _ = make_hierarchy_from_edges([(labels[int(parent)], labels[int(child)]) for parent, child in edges])

    specificity = -tree.num_leaf_descendants()

    state = torch.load(f"{models_dir}/{model_name}/latest.pth")
    model = ClassiaImageModelV1(tree)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    transform = Compose([
        CenterCrop(224),
        ToImageTensor(),
        ConvertImageDtype(torch.float),
        Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.244, 0.225]
        )
    ])

    with torch.no_grad():

        for i in range(0, len(files), batch_size):
            images = []
            for file in files[i:i + batch_size]:
                with Image.open(file, 'r') as f:
                    image = transform(f)
                    images.append(image)

            x = torch.stack(images)
            x = x.to(device)
            theta = model(x)
            prob = SumDescendants(tree, strict=False).to(device)(softmax(theta, dim=-1), dim=-1).cpu()

            pred_idxs = [
                pareto_optimal_predictions(specificity, p, None, None)
                for p in prob
            ]
            pred_paths = [
                "/".join(labels[i] for i in np.nonzero(seq)[0])
                for seq in pred_idxs
            ]

            for path in pred_paths:
                print(path)


def predict_docs(files, *, models_dir, model_name, batch_size=8, device='cuda'):
    with open(f"{models_dir}/{model_name}/labels.txt", "r") as f:
        lines = [line.strip() for line in f.readlines()]
        labels = lines
    with open(f"{models_dir}/{model_name}/tree.csv", "r") as f:
        lines = [line.strip() for line in f.readlines()]
        edges = [line.split(",") for line in lines]
        tree, _ = make_hierarchy_from_edges([(labels[int(parent)], labels[int(child)]) for parent, child in edges])

    is_leaf = tree.leaf_mask()
    specificity = -tree.num_leaf_descendants()
    not_trivial = (tree.num_children() != 1)

    state = torch.load(f"{models_dir}/{model_name}/latest.pth")
    model = ClassiaTextModelV1(tree)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    transform = ROBERTA_BASE_ENCODER.transform()

    with torch.no_grad():

        for i in range(0, len(files), batch_size):
            texts = []
            for file in files[i:i + batch_size]:
                with open(file, 'r') as f:
                    text = f.read()
                    texts.append(text)

            x = torch.tensor(transform(texts))
            x = x.to(device)
            theta = model(x)
            prob = SumDescendants(tree, strict=False).to(device)(softmax(theta, dim=-1), dim=-1).cpu().numpy()

            pred_idxs = [
                pareto_optimal_predictions(specificity, p, MIN_THRESHOLD, not_trivial)
                for p in prob
            ]

            pred_paths = [
                "/".join(labels[i] for i in seq if i!=0) # exclude root (0)
                for seq in pred_idxs 
            ]

            for path in pred_paths:
                print(path)


def pareto_optimal_predictions(
        info: np.ndarray,
        prob: np.ndarray,
        min_threshold: Optional[float] = None,
        condition: Optional[np.ndarray] = None,
        require_unique: bool = False,
) -> np.ndarray:
    """Finds the sequence of nodes that can be chosen by threshold.

    Returns nodes that are more specific than all more-confident predictions.
    This is equivalent to:
    (1) nodes such that there does not exist a node which is more confident and more specific,
    (2) nodes such that all nodes are less confident or less specific (*less than or equal).

    The resulting nodes are ordered descending by prob (and ascending by info).
    """
    assert prob.ndim == 1
    assert info.ndim == 1

    is_valid = np.ones(prob.shape, dtype=bool)
    if min_threshold is not None:
        is_valid = is_valid & (prob > min_threshold)
    if condition is not None:
        is_valid = is_valid & condition
    assert np.any(is_valid), 'require at least one valid element'
    prob = prob[is_valid]
    info = info[is_valid]
    valid_inds, = np.nonzero(is_valid)

    # Order descending by prob then descending by info.
    # Note that np.lexsort() orders first by the last key.
    # (Performs stable sort from first key to last key.)
    order = np.lexsort((-info, -prob))
    prob = prob[order]
    info = info[order]

    max_info = np.maximum.accumulate(info)
    keep = (prob[1:] > prob[:-1]) | (info[1:] > max_info[:-1])
    keep = np.concatenate(([True], keep))

    if require_unique:
        if np.any((prob[1:] == prob[:-1]) & (info[1:] == info[:-1]) & (keep[1:] | keep[:-1])):
            raise ValueError('set is not unique')

    return valid_inds[order[keep]]
