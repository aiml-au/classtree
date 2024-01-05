import os
from pathlib import Path

import torch
from torch.utils.data import Dataset

from PIL import Image

from .hier import make_hierarchy_from_edges


class ClasstreeImageDataset(Dataset):
    def __init__(self, files, labels, transform):
        self.files = files
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        with Image.open(self.files[i]) as image:
            image = image.convert("RGB")
            image = self.transform(image)

            return image, self.labels[i]


class ClasstreeTextDataset(Dataset):
    def __init__(self, files, labels, *, transform):
        self.files = files
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        with open(self.files[i], "r") as f:
            text = f.read()
            input_ids = torch.tensor(self.transform(text), dtype=torch.long)
            attention_mask = torch.ones_like(input_ids)

            return input_ids, attention_mask, self.labels[i]


def hierarchy_and_labels_from_folder(folder):
    folder = Path(folder).expanduser()

    file_list = []
    labels_list = []
    edge_set = set()
    edge_list = []

    for root, dirs, files in os.walk(folder):
        # HACK: os.walk doesn't guarantee order, so we mutate
        # the dirs list that will be used internally to enforce an order
        dirs.sort()
        files.sort()
        for file in files:
            file_path = Path(root).joinpath(Path(file))
            file_classes = file_path.parent.relative_to(folder.parent).parts
            file_list.append(file_path)
            labels_list.append(file_classes)

            for i in range(1, len(file_classes)):
                parent = file_classes[i - 1]
                child = file_classes[i]
                edge = (parent, child)
                if edge not in edge_set:
                    edge_set.add(edge)
                    edge_list.append(edge)

    tree, node_labels = make_hierarchy_from_edges(edge_list)
    label_indices = {label: i for i, label in enumerate(node_labels)}

    return (
        tree,
        node_labels,
        file_list,
        [label_indices[label_parts[-1]] for label_parts in labels_list],
    )
