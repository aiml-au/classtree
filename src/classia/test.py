import logging
import os

import torch

from classia.dataset import hierarchy_and_labels_from_folder
from classia.models import model_classes, get_text_encoder
from classia.train import get_dataloader, evaluate, get_image_dataset, get_text_dataset

LOGGER = logging.getLogger(__name__)


def test_model(models_dir, model_name, batch_size, dir, device):
    if not os.path.exists(f"{models_dir}/{model_name}/best.pth"):
        raise FileNotFoundError(
            f"Model {model_name} does not exist ({models_dir}/{model_name})"
        )

    checkpoint = torch.load(f"{models_dir}/{model_name}/best.pth")
    model_id = checkpoint["model_id"]
    model_type = checkpoint["model_type"]
    model_size = checkpoint["model_size"]
    tree = checkpoint["tree"]

    model_class = model_classes[model_id]
    model = model_class(tree)
    model.load_state_dict(checkpoint["model_state_dict"])

    tree, _, files, labels = hierarchy_and_labels_from_folder(dir)

    if model_type == "image":
        eval_dataset = get_image_dataset(files, labels)
    elif model_type == "text":
        encoder = get_text_encoder(model_size)
        eval_dataset = get_text_dataset(files, labels, encoder=encoder)
    else:
        raise ValueError(f"Unknown model type {model_type}")

    eval_loader = get_dataloader(
        model_type, eval_dataset, batch_size=batch_size, model_size=model_size
    )
    evaluate(model, eval_loader, tree)
