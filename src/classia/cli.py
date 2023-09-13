import argparse
import logging
import json
import os

from classia.dataset import hierarchy_and_labels_from_folder
from .train import train_model, write_data, get_dataset, get_dataloader, prepare_dataloaders, evaluate
from .predict import predict_images, predict_docs
from .models import get_text_model, get_image_model

from fs import open_fs
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

REMOTE_FS_URL = os.getenv("REMOTE_FS", default="gs://aiml-shop-classia-public") # /models

def run():
    global REMOTE_FS_URL

    parser = argparse.ArgumentParser(description="A hierarchical classifier")

    parser.add_argument("--datasets_dir", help="The path of the directory to store named datasets",
                        default=os.path.expanduser("~/.cache/classia/datasets"))
    parser.add_argument("--models_dir", help="The path of the directory to store named models",
                        default=os.path.expanduser("~/.cache/classia/models"))
    
    parser.add_argument("--image_model_size", help="The size variant for the Image model: (small, medium, large)", default="medium")
    parser.add_argument("--text_model_size", help="The size variant for the Text model: (base, large)", default="base")

    subparsers = parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser("train", help="Train a new hierarchical classification model")
    train_parser.add_argument("--model", help="The name of the model", type=str, required=True)
    train_parser.add_argument("--epochs", help="The number of epochs to train for", type=int, default=10)
    train_parser.add_argument("--batch_size", help="The batch size to use during training", type=int, default=8)
    train_parser.add_argument("--lr", help="The learning rate to use during training", type=float, default=0.001)
    train_parser.add_argument("--resume", help="Resume training from the last best epoch", type=bool, default=False)
    train_datasource_group = train_parser.add_mutually_exclusive_group()
    train_datasource_group.add_argument("--images",
                                        help="The directory of training images, if training an image classifier")
    train_datasource_group.add_argument("--docs",
                                        help="The directory of training documents, if training a text classifier")

    test_parser = subparsers.add_parser("test", help="Evaluate a hierarchical classification model using a test set")
    test_parser.add_argument("--model", help="The name of the model", required=True)
    test_parser.add_argument("--batch_size", help="The batch size to use for evaluation dataloaders", type=int, default=8)
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
    download_parser.add_argument("--dataset", help="The name of the dataset")
    download_parser.add_argument("--download_dir", help="The path of the directory to download named models",
                        default=os.path.expanduser("~/.cache/classia"))

    args = parser.parse_args()

    if not args.command:
        parser.print_help()

    elif args.command == "train":        
        os.makedirs(f'{args.models_dir}/{args.model}', exist_ok=True)
        model_type = None

        if args.images:           
            model_type = 'image'
            model_size = args.image_model_size            
            tree, label_set, files, labels = hierarchy_and_labels_from_folder(args.images)
            model= get_image_model(tree, args.image_model_size)
        elif args.docs:
            model_type = 'text'
            model_size = args.text_model_size
            tree, label_set, files, labels = hierarchy_and_labels_from_folder(args.docs)
            model= get_text_model(tree, args.text_model_size)
        else:
            train_parser.error("One of --images or --docs must be provided.")

        if model_type:       
            meta_data = {"model_type": model_type, "model_size": model_size, "model_weights": model._get_name()}
            write_data(tree, label_set, files, labels, meta_data, models_dir=args.models_dir, model_name=args.model)
            dataset = get_dataset(model_type, files, labels, model_size=model_size)
            train_loader, eval_loader = prepare_dataloaders(model_type, dataset, batch_size=args.batch_size, model_size=model_size)
            train_model(model, train_loader, eval_loader, tree, label_set, models_dir=args.models_dir, model_name=args.model, model_size=model_size, epochs=args.epochs, lr=args.lr, resume=args.resume)

    elif args.command == "test":        
        if args.images:
            LOGGER.info(f"Test {args.model} on {args.images}")
            model_type = 'image'
            model_size = args.image_model_size
            tree, _ ,files, labels = hierarchy_and_labels_from_folder(args.images)
            model= get_image_model(tree, args.image_model_size)
        elif args.docs:
            LOGGER.info(f"Test {args.model} on {args.docs}")
            model_type = 'text'
            model_size = args.text_model_size
            tree, _ ,files, labels = hierarchy_and_labels_from_folder(args.docs)
            model= get_text_model(tree, args.text_model_size)
        else:
            train_parser.error("One of --images or --docs must be provided.")

        eval_dataset = get_dataset(model_type, files, labels, model_size=model_size)
        eval_loader = get_dataloader(model_type, eval_dataset, batch_size=args.batch_size, model_size=model_size)
        evaluate(model, eval_loader, tree, models_dir=args.models_dir, model_name=args.model)

    elif args.command == "predict":

        if not os.path.exists(f"{args.models_dir}/{args.model}/meta.json"):
            predict_parser.error(f"Model {args.model} does not exist ({args.models_dir}/{args.model})")
        else:
            with open(f"{args.models_dir}/{args.model}/meta.json", "r") as f:
                meta = json.load(f)

            if meta["model_type"] == "image":
                predict_images(args.files, models_dir=args.models_dir, model_name=args.model)
            elif meta["model_type"] == "text":
                predict_docs(args.files, models_dir=args.models_dir, model_name=args.model)
            else:
                predict_parser.error(f"Unknown model type {meta['type']}")

    elif args.command == ("download"):

        if args.model:   
            download_dir = f"{args.download_dir}/models/{args.model}" if ".cache" in args.download_dir else f"{args.download_dir}/{args.model}"
            REMOTE_FS_URL += f"/models/{args.model}" 
            model_weights = "best.pth" # ["best.pth", "latest.pth"]
            download('Model', REMOTE_FS_URL, model_weights, download_dir)

        elif args.dataset:
            download_dir = f"{args.download_dir}/datasets/{args.dataset}" if ".cache" in args.download_dir else f"{args.download_dir}/{args.dataset}"
            REMOTE_FS_URL += f"/datasets"
            dataset_name = args.dataset if ".zip" in args.dataset else f"{args.dataset}.zip"
            download('Dataset', REMOTE_FS_URL, dataset_name, download_dir)
        else:            
            download_parser.error("One of --model or --dataset must be provided.")


def download(flag, remote_fs_url, file, download_dir):            
    fs = open_fs(remote_fs_url)

    if not fs.exists(file):
        LOGGER.warning(
            f"{flag}: {file} does not exist in the remote location."
        )
    else:          
        os.makedirs(download_dir, exist_ok=True)
        loca_target_file = f'{download_dir}/{file}'

        if not os.path.exists(loca_target_file):
            LOGGER.info(f"Downloading {file}")
        else:
            LOGGER.warning(f"Overriding {file} in {download_dir} by downloading from remote location.")

        with open(loca_target_file, "wb") as f:
            fs.download(file, f) 

        LOGGER.info(f'{flag} was successfully downloaded in {loca_target_file}')
        
