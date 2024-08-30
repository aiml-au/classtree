import argparse
import logging
import os
import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")


from .dataset import hierarchy_and_labels_from_folder
from .download import download_model, download_text_dataset, download_image_dataset
from .export import export_model
from .test import test_model
from .train import (
    train_image_model,
    train_text_model,
)
from .predict import predict

import torch

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


def add_global_args(parser):
    parser.add_argument(
        "--datasets_dir",
        help="The path of the directory to store named datasets",
        default=os.path.expanduser("~/.cache/classtree/datasets"),
    )
    parser.add_argument(
        "--models_dir",
        help="The path of the directory to store named models",
        default=os.path.expanduser("~/.cache/classtree/models"),
    )


def add_train_args(train_parser):
    add_global_args(train_parser)
    train_parser.add_argument(
        "--device",
        help="The device to train on",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    train_parser.add_argument(
        "--epochs", help="The number of epochs to train for", type=int, default=10
    )
    train_parser.add_argument(
        "--batch_size",
        help="The batch size to use during training",
        type=int,
        default=64,
    )
    train_parser.add_argument(
        "--lr",
        help="The learning rate to use during training",
        type=float,
        default=0.001,
    )
    base_model_group = train_parser.add_mutually_exclusive_group(required=False)
    base_model_group.add_argument(
        "--from",
        help="Load another model to use for transfer learning/fine-tuning.",
        dest="from_model",
        type=str,
        default=None,
    )
    base_model_group.add_argument(
        "--resume",
        help="Resume training from the last epoch",
        action="store_true",
    )


def add_train_text_args(train_text_parser):
    add_train_args(train_text_parser)
    train_text_parser.add_argument(
        "--size", help="The size of the model: (m/l)", choices=["m", "l"], default="m"
    )
    train_text_parser.add_argument(
        "--model", help="The name of the model", type=str, required=True
    )
    train_text_parser.add_argument(
        "--dir", help="The path to a folder of text files", type=str, required=True
    )


def add_train_image_args(train_image_parser):
    add_train_args(train_image_parser)
    train_image_parser.add_argument(
        "--size",
        help="The size of the model: (s/m/l)",
        choices=["s", "m", "l"],
        default="m",
    )
    train_image_parser.add_argument(
        "--model", help="The name of the model", type=str, required=True
    )
    train_image_parser.add_argument(
        "--dir", help="The path to a folder of images", type=str, required=True
    )


def add_test_args(test_parser):
    add_global_args(test_parser)
    test_parser.add_argument(
        "--device",
        help="The device to infer on",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    test_parser.add_argument("--model", help="The name of the model", required=True)
    test_parser.add_argument(
        "--batch_size",
        help="The batch size to use for evaluation dataloaders",
        type=int,
        default=8,
    )
    test_parser.add_argument(
        "--dir",
        help="The directory of testing documents, if testing a text classifier",
    )


def add_predict_args(predict_parser):
    add_global_args(predict_parser)
    predict_parser.add_argument(
        "--device",
        help="The device to infer on",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    predict_parser.add_argument("--model", help="The name of the model", required=True)
    predict_parser.add_argument(
        "--batch_size",
        help="The batch size to use for evaluation dataloaders",
        type=int,
        default=8,
    )
    predict_parser.add_argument(
        "files", nargs="+", help="The filenames of the examples to classify."
    )


def add_download_args(download_parser):
    add_global_args(download_parser)
    download_target_group = download_parser.add_mutually_exclusive_group()
    download_target_group.add_argument("--model", help="The name of the model")
    download_target_group.add_argument("--images", help="The name of an image dataset")
    download_target_group.add_argument("--text", help="The name of a text dataset")


def add_export_args(export_parser):
    add_global_args(export_parser)
    export_parser.add_argument("--model", help="The name of the model")
    export_parser.add_argument(
        "--export_dir",
        help="The path of the directory to export models",
        default=os.path.expanduser("~/.cache/classtree"),
    )
    export_parser.add_argument(
        "--device",
        help="The device to export on",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )


def run():
    parser = argparse.ArgumentParser(description="A hierarchical classifier")

    add_global_args(parser)

    subparsers = parser.add_subparsers(dest="command")
    train_parser = subparsers.add_parser(
        "train", help="Train a new hierarchical classification model"
    )
    add_train_args(train_parser)

    train_subparsers = train_parser.add_subparsers(dest="type")

    train_text_parser = train_subparsers.add_parser(
        "text", help="Train a hierarchical text classifier"
    )
    add_train_text_args(train_text_parser)

    train_image_parser = train_subparsers.add_parser(
        "images", help="Train a hierarchical image classifier"
    )
    add_train_image_args(train_image_parser)

    test_parser = subparsers.add_parser(
        "test", help="Evaluate a hierarchical classification model using a test set"
    )
    add_test_args(test_parser)

    predict_parser = subparsers.add_parser(
        "predict",
        help="Use a trained hierarchical classification model on unlabelled examples",
    )
    add_predict_args(predict_parser)

    download_parser = subparsers.add_parser(
        "download", help="Download pre-trained model weights"
    )
    add_download_args(download_parser)

    export_parser = subparsers.add_parser(
        "export", help="Export pre-trained model weights"
    )
    add_export_args(export_parser)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()

    elif args.command == "train":
        if args.type == "images":
            tree, label_set, files, labels = hierarchy_and_labels_from_folder(args.dir)

            train_image_model(
                models_dir=args.models_dir,
                model_name=args.model,
                model_size=args.size,
                tree=tree,
                label_set=label_set,
                files=files,
                labels=labels,
                batch_size=args.batch_size,
                epochs=args.epochs,
                lr=args.lr,
                resume=args.resume,
                from_model_name=args.from_model,
                device=args.device,
            )

        elif args.type == "text":
            tree, label_set, files, labels = hierarchy_and_labels_from_folder(args.dir)

            train_text_model(
                models_dir=args.models_dir,
                model_name=args.model,
                model_size=args.size,
                tree=tree,
                label_set=label_set,
                files=files,
                labels=labels,
                batch_size=args.batch_size,
                epochs=args.epochs,
                lr=args.lr,
                resume=args.resume,
                from_model_name=args.from_model,
                device=args.device,
            )
        else:
            train_parser.print_help()

    elif args.command == "test":
        test_model(args.models_dir, args.model, args.batch_size, args.dir, args.device)

    elif args.command == "predict":
        if not os.path.exists(f"{args.models_dir}/{args.model}/best.pth"):
            predict_parser.error(
                f"Model {args.model} does not exist ({args.models_dir}/{args.model})"
            )

        predict(
            models_dir=args.models_dir,
            model_name=args.model,
            files=args.files,
            batch_size=args.batch_size,
            device=args.device,
        )

    elif args.command == "download":
        if args.model:
            download_model(args.model, args.models_dir)
        elif args.text:
            download_text_dataset(args.text, args.datasets_dir)
        elif args.images:
            download_image_dataset(args.images, args.datasets_dir)
        else:
            download_parser.error("One of --model or --dataset must be provided.")

    elif args.command == "export":
        export_model(args.models_dir, args.model, args.device)
