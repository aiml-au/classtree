import argparse
import json
import os

from classia.dataset import hierarchy_and_labels_from_folder

from .train import train_image_model, train_text_model
from .predict import predict_images, predict_docs


def run():
    parser = argparse.ArgumentParser(description="A hierarchical classifier")

    parser.add_argument("--datasets_dir", help="The path of the directory to store named datasets",
                        default=os.path.expanduser("~/.cache/classia/datasets"))
    parser.add_argument("--models_dir", help="The path of the directory to store named models",
                        default=os.path.expanduser("~/.cache/classia/models"))

    subparsers = parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser("train", help="Train a new hierarchical classification model")
    train_parser.add_argument("--model", help="The name of the model", required=True)
    train_parser.add_argument("--epochs", help="The number of epochs to train for", default=10)
    train_parser.add_argument("--batch_size", help="The batch size to use during training", default=8)
    train_parser.add_argument("--lr", help="The learning rate to use during training", default=0.001)
    train_datasource_group = train_parser.add_mutually_exclusive_group()
    train_datasource_group.add_argument("--images",
                                        help="The directory of training images, if training an image classifier")
    train_datasource_group.add_argument("--docs",
                                        help="The directory of training documents, if training a text classifier")

    test_parser = subparsers.add_parser("test", help="Evaluate a hierarchical classification model using a test set")
    test_parser.add_argument("--model", help="The name of the model", required=True)
    test_datasource_group = test_parser.add_mutually_exclusive_group()
    test_datasource_group.add_argument("--images",
                                       help="The directory of testing images, if testing an image classifier")
    test_datasource_group.add_argument("--docs",
                                       help="The directory of testing documents, if testing a text classifier")

    predict_parser = subparsers.add_parser("predict",
                                           help="Use a trained hierarchical classification model on unlabelled examples")
    predict_parser.add_argument("--model", help="The name of the model", required=True)
    predict_parser.add_argument("files", nargs="+", help="The filenames of the examples to classify.")

    download_parser = subparsers.add_parser("download", help="Download pre-trained model weights")
    download_parser.add_argument("--model", help="The name of the model")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()

    elif args.command == "train":

        if args.images:
            train_image_model(args.images, models_dir=args.models_dir, model_name=args.model, epochs=args.epochs,
                              batch_size=args.batch_size, lr=args.lr)
        elif args.docs:
            train_text_model(args.docs, models_dir=args.models_dir, model_name=args.model, epochs=args.epochs,
                             batch_size=args.batch_size, lr=args.lr)
        else:
            train_parser.error("One of --images or --docs must be provided.")

    elif args.command == "test":
        if args.images:
            print(f"Test {args.model} on {args.images}")
            dirs, files = hierarchy_and_labels_from_folder(args.images)
            print(dirs)
            print(files)
        elif args.docs:
            print(f"Test {args.model} on {args.docs}")
            dirs, files = hierarchy_and_labels_from_folder(args.docs)
            print(dirs)
            print(files)
        else:
            train_parser.error("One of --images or --docs must be provided.")

    elif args.command == "predict":

        if not os.path.exists(f"{args.models_dir}/{args.model}/meta.json"):
            predict_parser.error(f"Model {args.model} does not exist ({args.models_dir}/{args.model})")
        else:
            with open(f"{args.models_dir}/{args.model}/meta.json", "r") as f:
                meta = json.load(f)

            if meta["type"] == "image":
                predict_images(args.files, models_dir=args.models_dir, model_name=args.model)
            elif meta["type"] == "text":
                predict_docs(args.files, models_dir=args.models_dir, model_name=args.model)
            else:
                predict_parser.error(f"Unknown model type {meta['type']}")

    elif args.command == ("download"):
        print(f"Download {args.model}")
