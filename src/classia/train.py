import logging
import json
import collections
import os.path

import numpy as np

import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import SGD
from torch.nn.functional import softmax
from torchvision.transforms.v2 import (
    Compose,
    RandomResizedCrop,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    RandomGrayscale,
    ToImageTensor,
    ConvertImageDtype,
    Normalize,
)

# from torchtext.models.roberta.bundler import ROBERTA_BASE_ENCODER
from .models import get_text_encoder, get_model_id, model_classes
import matplotlib.pyplot as plt
from tqdm import tqdm

from .hier import truncate_given_lca, FindLCA, SumDescendants
from .dataset import ClassiaImageDataset, ClassiaTextDataset
from .loss import MarginLoss
from .metrics import UniformLeafInfoMetric, DepthMetric, IsCorrect, operating_curve
from .predict import pareto_optimal_predictions

LOGGER = logging.getLogger(__name__)

PATIENCE = 10
SPLIT_RATIO = 0.8
MIN_THRESHOLD = 0.5


def convert(o):
    if isinstance(o, np.int64):
        return int(o)
    raise TypeError


def get_image_dataset(files, labels):
    transform = Compose(
        [
            RandomResizedCrop(224),
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
            RandomGrayscale(),
            ToImageTensor(),
            ConvertImageDtype(torch.float),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.244, 0.225]),
        ]
    )
    dataset = ClassiaImageDataset(files, labels, transform=transform)
    return dataset


def get_text_dataset(files, labels, encoder):
    transform = encoder.transform()
    dataset = ClassiaTextDataset(files, labels, transform=transform)
    return dataset


def get_dataloader(model_type, dataset, batch_size=8, model_size="base"):
    def pad_tokens(batch):
        max_tokens = max((seq.shape[0] for seq, _, _ in batch))

        input_ids = torch.full(
            (len(batch), max_tokens),
            fill_value=get_text_encoder(model_size).encoderConf.padding_idx,
        )
        for i in range(len(batch)):
            input_ids[i, : len(batch[i][0])] = batch[i][0]

        labels = torch.tensor([label for _, _, label in batch], dtype=torch.long)

        return input_ids, labels

    collate_fn = {"collate_fn": pad_tokens} if model_type == "text" else {}

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, **collate_fn)

    return data_loader


def prepare_dataloaders(model_type, dataset, batch_size=8, model_size="base"):
    # TODO: Stratified splits so that we don't accidentally drop entire classes.
    train_dataset, eval_dataset = random_split(dataset, (SPLIT_RATIO, 1 - SPLIT_RATIO))

    train_loader = get_dataloader(model_type, train_dataset, batch_size, model_size)
    eval_loader = get_dataloader(model_type, eval_dataset, batch_size, model_size)
    return train_loader, eval_loader


# Save training/validation loss plots
def save_training_plot(train_loss, val_loss, save_dir):
    LOGGER.info("Saving loss plots...")

    plt.figure(figsize=(10, 7))
    plt.plot(train_loss, color="orange", label="train loss")
    plt.plot(val_loss, color="red", label="validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"{save_dir}/loss_plot.png")


LabelMap = collections.namedtuple("LabelMap", ["to_node", "to_target"])


def get_metric_fns(tree):
    info_metric = UniformLeafInfoMetric(tree)
    depth_metric = DepthMetric(tree)
    metric_fns = {
        "exact": lambda gt, pr: pr == gt,
        "correct": IsCorrect(tree),
        "info_recall": info_metric.recall,
        "info_precision": info_metric.precision,
        "depth_recall": depth_metric.recall,
        "depth_precision": depth_metric.precision,
    }
    return metric_fns


def get_label_map(tree):
    label_map = LabelMap(
        to_node=np.arange(tree.num_nodes()), to_target=np.arange(tree.num_nodes())
    )  # to_target=tree.leaf_subset() when training only on leaf node

    # Convert target map to torch tensor.
    label_map = LabelMap(
        to_node=label_map.to_node, to_target=torch.from_numpy(label_map.to_target)
    )

    return label_map


def load_state(models_dir, model_name, model_type, model_size, from_model_name, resume):
    state = None
    init_epoch = 0
    init_best_loss = float("inf")

    if from_model_name:
        existing_model_path = f"{models_dir}/{from_model_name}/best.pth"
    elif resume:
        existing_model_path = f"{models_dir}/{model_name}/latest.pth"
    else:
        if os.path.exists(f"{models_dir}/{model_name}"):
            raise ValueError(
                f"Model {model_name} already exists. Set --resume to resume training."
            )
        existing_model_path = None

    if existing_model_path:
        if not os.path.exists(existing_model_path):
            raise FileNotFoundError(existing_model_path)
        else:
            checkpoint = torch.load(existing_model_path)
            if "model_type" in checkpoint and model_type != checkpoint["model_type"]:
                raise ValueError(
                    f"Model type mismatch: {model_type} != {checkpoint['model_type']}"
                )

            if "model_size" in checkpoint and model_size != checkpoint["model_size"]:
                raise ValueError(
                    f"Model size mismatch: {model_size} != {checkpoint['model_size']}"
                )

        state = checkpoint["model_state_dict"]
        if resume:
            init_epoch = checkpoint["epoch"] if "epoch" in checkpoint else 0
            init_best_loss = (
                checkpoint["loss"] if "loss" in checkpoint else float("inf")
            )

    return state, init_epoch, init_best_loss


def train_image_model(
    *,
    models_dir,
    model_name,
    model_size,
    tree,
    label_set,
    files,
    labels,
    batch_size,
    epochs,
    lr,
    resume,
    from_model_name,
    device,
):
    model_type = "image"
    model_id = get_model_id(model_type, 1, model_size)
    model_class = model_classes[model_id]
    model = model_class(tree)

    state, start_epoch, init_best_loss = load_state(
        models_dir, model_name, model_type, model_size, from_model_name, resume
    )
    if state:
        model.load_state_dict(state)

    os.makedirs(f"{models_dir}/{model_name}", exist_ok=True)

    dataset = get_image_dataset(files, labels)
    train_loader, eval_loader = prepare_dataloaders(
        model_type, dataset, batch_size=batch_size, model_size=model_size
    )
    train_model(
        model,
        train_loader,
        eval_loader,
        tree,
        label_set,
        models_dir=models_dir,
        model_name=model_name,
        model_type=model_type,
        model_size=model_size,
        epochs=epochs,
        start_epoch=start_epoch,
        best_loss=init_best_loss,
        lr=lr,
        device=device,
    )


def train_text_model(
    *,
    models_dir,
    model_name,
    model_size,
    tree,
    label_set,
    files,
    labels,
    batch_size,
    epochs,
    lr,
    resume,
    from_model_name,
    device,
):
    model_type = "text"
    model_id = get_model_id(model_type, 1, model_size)
    model_class = model_classes[model_id]
    encoder = get_text_encoder(model_size)
    model = model_class(tree)

    state, start_epoch, init_best_loss = load_state(
        models_dir, model_name, model_type, model_size, from_model_name, resume
    )
    if state:
        model.load_state_dict(state)

    os.makedirs(f"{models_dir}/{model_name}", exist_ok=True)

    dataset = get_text_dataset(files, labels, encoder)
    train_loader, eval_loader = prepare_dataloaders(
        model_type, dataset, batch_size=batch_size, model_size=model_size
    )
    train_model(
        model,
        train_loader,
        eval_loader,
        tree,
        label_set,
        models_dir=models_dir,
        model_name=model_name,
        model_type=model_type,
        model_size=model_size,
        epochs=epochs,
        start_epoch=start_epoch,
        best_loss=init_best_loss,
        lr=lr,
        device=device,
    )


def train_model(
    model,
    train_loader,
    eval_loader,
    tree,
    label_set,
    models_dir,
    model_name,
    model_type,
    model_size,
    epochs=50,
    start_epoch=0,
    best_loss=float("inf"),
    lr=0.001,
    device="cuda",
):
    train_label_map = eval_label_map = get_label_map(tree)

    model.to(device)
    loss_fn = MarginLoss(tree, with_leaf_targets=False, device=device)

    optimizer = SGD(model.parameters(), lr=lr)

    best_epoch = start_epoch
    train_loss_history, eval_loss_history = [], []

    items_to_log = dict.fromkeys(["epoch", "eval_loss", "train_loss"])
    training_logs = {"best": items_to_log, "logs": items_to_log}

    # Bundle hierarchy and metadata
    model_dict = {
        "labels": label_set,
        "tree": tree,
        "model_id": get_model_id(model_type, 1, model_size),
        "model_type": model_type,
        "model_size": model_size,
        "model_weight_name": model._get_name(),
    }

    end_epoch = start_epoch + epochs

    for epoch in range(start_epoch + 1, end_epoch + 1):
        total_train_loss = 0.0
        model.train()

        progress = tqdm(
            train_loader, desc=f"Train Epoch {epoch}/{end_epoch}", position=1
        )
        for inputs, gt_labels in progress:
            inputs = inputs.to(device)

            gt_targets = train_label_map.to_target[gt_labels]
            assert torch.all(gt_targets >= 0)
            optimizer.zero_grad()
            theta = model(inputs)

            loss = loss_fn(theta, gt_targets.to(device))

            loss.backward()
            optimizer.step()

            train_loss = loss.item()
            total_train_loss += train_loss

            progress.set_postfix({"loss": f"{train_loss:.4f}"})

        mean_train_loss = total_train_loss / len(train_loader)
        train_loss_history.append(mean_train_loss)

        # Validate the model
        with torch.no_grad():
            model.eval()
            total_eval_loss = 0.0

            for inputs, gt_labels in tqdm(
                eval_loader, desc=f"Eval Epoch {epoch}/{end_epoch}", position=1
            ):
                inputs = inputs.to(device)
                theta = model(inputs)

                gt_targets = eval_label_map.to_target[gt_labels]
                assert torch.all(gt_targets >= 0)

                loss = loss_fn(theta, gt_targets.to(device))
                eval_loss = loss.item()
                total_eval_loss += eval_loss

        mean_eval_loss = total_eval_loss / len(eval_loader)
        eval_loss_history.append(mean_eval_loss)

        LOGGER.info(
            "Train Loss: {} | Eval Loss: {} at Epoch:{}".format(
                mean_train_loss, mean_eval_loss, epoch
            )
        )

        model_dict["model_state_dict"] = model.state_dict()
        model_dict["epoch"] = epoch
        model_dict["loss"] = mean_eval_loss

        # Save latest model
        torch.save(model_dict, f"{models_dir}/{model_name}/latest.pth")

        # Save best model at best evaluation loss
        if mean_eval_loss < best_loss:
            best_loss = mean_eval_loss
            best_epoch = epoch

            training_logs["best"]["epoch"] = best_epoch
            training_logs["best"]["eval_loss"] = mean_eval_loss
            training_logs["best"]["train_loss"] = mean_train_loss

            LOGGER.info(
                f"Saving best model at Epoch {best_epoch} | val_loss: {best_loss}"
            )

            model_dict["best_epoch"] = best_epoch
            model_dict["best_loss"] = best_loss

            torch.save(model_dict, f"{models_dir}/{model_name}/best.pth")

        training_logs["logs"]["epoch"] = epoch
        training_logs["logs"]["eval_loss"] = mean_eval_loss
        training_logs["logs"]["train_loss"] = mean_train_loss

        # Early stopping
        if best_epoch <= epoch - PATIENCE:  # nothing is improving for a while
            LOGGER.info(
                f"Early stopping.. at Epoch {epoch} with patience of {PATIENCE} epochs"
            )
            break

    LOGGER.info("TRAINING COMPLETE")
    save_training_plot(
        train_loss_history, eval_loss_history, f"{models_dir}/{model_name}"
    )

    with open(f"{models_dir}/{model_name}/logs.json", "w") as f:
        json.dump(training_logs, f, indent=4)


def evaluate(model, eval_loader, tree, device="cuda"):
    model.to(device)

    specificity = -tree.num_leaf_descendants()
    not_trivial = tree.num_children() != 1

    eval_label_map = get_label_map(tree)

    gt = []
    seq_outputs_prob, seq_outputs_pred = [], []

    with torch.no_grad():
        model.eval()

        for inputs, gt_labels in tqdm(eval_loader, position=1):
            inputs = inputs.to(device)
            theta = model(inputs)

            def pred_fn(theta):
                return SumDescendants(tree, strict=False).to(device)(
                    softmax(theta, dim=-1), dim=-1
                )

            prob = pred_fn(theta).cpu().numpy()
            gt_node = eval_label_map.to_node[gt_labels]

            pred_seqs = [
                pareto_optimal_predictions(specificity, p, MIN_THRESHOLD, not_trivial)
                for p in prob
            ]
            prob_seqs = [prob[i, pred_i] for i, pred_i in enumerate(pred_seqs)]

            # Concatenate array results from minibatches.
            gt.extend(gt_node)
            seq_outputs_pred.extend(pred_seqs)
            seq_outputs_prob.extend(prob_seqs)

    pareto_means = assess_predictions(tree, gt, seq_outputs_prob, seq_outputs_pred)

    LOGGER.info("[Evaluation Metric]:")
    for metric, values in pareto_means.items():
        metric_values_means = np.mean(values)
        LOGGER.info(f'\t{metric}: {"{:.2f}".format(metric_values_means)}')


# Show metrics: from eval_inat21mini.ipynb
def assess_predictions(tree, gt, prob_seq, pred_seq):
    metric_fns = get_metric_fns(tree)

    # Evaluate predictions in Pareto sequence.
    find_lca = FindLCA(tree)
    pred_seq = [
        truncate_given_lca(gt_i, pr_i, find_lca(gt_i, pr_i))
        for gt_i, pr_i in zip(gt, pred_seq)
    ]
    metric_values_seq = {
        field: [metric_fn(gt_i, pr_i) for gt_i, pr_i in zip(gt, pred_seq)]
        for field, metric_fn in metric_fns.items()
    }

    _, pareto_totals = operating_curve(prob_seq, metric_values_seq)

    pareto_means = {k: v / len(gt) for k, v in pareto_totals.items()}

    return pareto_means
