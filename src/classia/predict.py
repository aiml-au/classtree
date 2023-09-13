from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch.nn.functional import softmax
from torchvision.transforms.v2 import CenterCrop, ToImageTensor, ConvertImageDtype, Normalize, Compose

from .hier import SumDescendants
from .models import get_image_model, get_text_model, get_text_encoder

MIN_THRESHOLD = 0.5


def predict_images(files, *, models_dir, model_name, batch_size=8, device='cuda'):

    checkpoint = torch.load(f"{models_dir}/{model_name}/best.pth")
    labels = checkpoint['labels']
    tree = checkpoint['tree']

    image_model_size = checkpoint['model_size']
    model= get_image_model(tree, image_model_size)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    specificity = -tree.num_leaf_descendants()
    not_trivial = (tree.num_children() != 1)

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


def predict_docs(files, *, models_dir, model_name, batch_size=8, device='cuda'):

    checkpoint = torch.load(f"{models_dir}/{model_name}/best.pth") 
    labels = checkpoint['labels']
    tree = checkpoint['tree']

    text_model_size = checkpoint['model_size']
    model= get_text_model(tree, text_model_size)
    text_encoder= get_text_encoder(text_model_size)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    specificity = -tree.num_leaf_descendants()
    not_trivial = (tree.num_children() != 1)

    transform = text_encoder.transform()
    padding_idx = 1 # for texts-embeddings for all instances to be in the same dimension
    import torchtext.functional as F

    with torch.no_grad():

        for i in range(0, len(files), batch_size):
            texts = []
            for file in files[i:i + batch_size]:
                with open(file, 'r') as f:
                    text = f.read()
                    texts.append(text)

            x = F.to_tensor(transform(texts), padding_value=padding_idx)
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


# Used in validation metrics
def argmax_with_confidence(
        value: np.ndarray,
        p: np.ndarray,
        threshold: float,
        condition: Optional[np.ndarray] = None) -> np.ndarray:
    """Finds element that maximizes (value, p) subject to p > threshold."""
    mask = (p > threshold)
    if condition is not None:
        mask = mask & condition
    return arglexmin_where(np.broadcast_arrays(-p, -value), mask)


from typing import Optional, Tuple
def arglexmin_where(
        keys: Tuple[np.ndarray, ...],
        condition: np.ndarray,
        axis: int = -1,
        keepdims: bool = False,
        ) -> np.ndarray:
    # TODO: Make more efficient (linear rather than log-linear).
    assert np.all(np.any(condition, axis=axis)), 'require at least one valid element'
    order = np.lexsort(keys, axis=axis)
    # Take first element in order that satisfies condition.
    # TODO: Would be faster to take subset and then sort?
    # Would this break the vectorization?
    # first_valid = np.argmax(np.take_along_axis(condition, order, axis=axis),
    #                         axis=axis, keepdims=True)
    first_valid = np.expand_dims(
        np.argmax(np.take_along_axis(condition, order, axis=axis), axis=axis),
        axis)
    result = np.take_along_axis(order, first_valid, axis=axis)
    if not keepdims:
        result = np.squeeze(result, axis=axis)
    return result
