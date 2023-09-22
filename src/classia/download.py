import os
from zipfile import ZipFile

from fsspec.core import get_fs_token_paths
import logging

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)
REMOTE_BASE = "gs://aiml-shop-classia-public"

model_urls = {
    "dbpedia": f"{REMOTE_BASE}/models/dbpedia.zip",
}

dataset_urls = {
    "dbpedia": f"{REMOTE_BASE}/datasets/dbpedia.zip",
    "inaturalist21-mini": f"{REMOTE_BASE}/datasets/inaturalist21-mini.zip",
}


def download_model(name, models_dir):
    model_url = model_urls[name]
    download(model_url, f"{models_dir}/{name}")


def download_dataset(name, datasets_dir):
    dataset_url = dataset_urls[name]
    download(dataset_url, f"{datasets_dir}/{name}")


def download(url, destination):
    zip_file = f"{destination}.zip"

    if os.path.exists(destination):
        return

    try:
        os.remove(zip_file)
    except FileNotFoundError:
        pass

    fs, _, [path] = get_fs_token_paths(url)
    fs.get_file(path, zip_file)

    LOGGER.info(f"Downloading data to {zip_file} complete.")

    with ZipFile(zip_file, "r") as zip_ref:
        zip_ref.extractall(destination)

    LOGGER.info(f"Extracting {zip_file} complete.")

    os.remove(zip_file)


def download_text_dataset(name, datasets_dir="datasets"):
    download_dataset(name, datasets_dir)


def download_image_dataset(name, datasets_dir="datasets"):
    download_dataset(name, datasets_dir)
