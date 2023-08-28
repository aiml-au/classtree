import json
import os

import torch

from torch.utils.data import DataLoader
from torch.optim import SGD
from torchvision.transforms.v2 import Compose, RandomResizedCrop, RandomHorizontalFlip, RandomVerticalFlip, \
    RandomGrayscale, ToImageTensor, ConvertImageDtype, Normalize
from torchtext.models.roberta.bundler import ROBERTA_BASE_ENCODER
from tqdm import tqdm

from .dataset import hierarchy_and_labels_from_folder, ClassiaImageDataset, ClassiaTextDataset
from .loss import MarginLoss
from .models import ClassiaImageModelV1, ClassiaTextModelV1


def train_image_model(folder, *, models_dir, model_name, epochs=50, batch_size=8, lr=0.001, device='cuda'):
    os.makedirs(f'{models_dir}/{model_name}', exist_ok=True)

    tree, label_set, files, labels = hierarchy_and_labels_from_folder(folder)

    # TODO: Save all of this in the .pth file with the model

    with open(f'{models_dir}/{model_name}/tree.csv', 'w') as f:
        f.write('\n'.join(f'{parent},{child}' for parent, child in tree.edges()))
    with open(f'{models_dir}/{model_name}/labels.txt', 'w') as f:
        f.write('\n'.join(label_set))
    with open(f'{models_dir}/{model_name}/train.csv', 'w') as f:
        f.write('\n'.join(f'{file},{label}' for file, label in zip(files, labels)))
    with open(f'{models_dir}/{model_name}/meta.json', 'w') as f:
        json.dump({'type': 'image'}, f)

    train_transform = Compose([
        RandomResizedCrop(224),
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        RandomGrayscale(),
        ToImageTensor(),
        ConvertImageDtype(torch.float),
        Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.244, 0.225]
        )
    ])

    dataset = ClassiaImageDataset(files, labels, transform=train_transform)

    model = ClassiaImageModelV1(tree)
    model.to(device)
    loss = MarginLoss(tree, with_leaf_targets=False)

    optimizer = SGD(model.parameters(), lr=lr)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()

    for epoch in tqdm(range(epochs), desc="Training", position=0):
        for x, y in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}", position=1):
            x = x.to(device)

            optimizer.zero_grad()
            theta = model(x).cpu()
            loss_value = loss(theta, y)
            loss_value.backward()
            optimizer.step()

        torch.save(model.state_dict(), f'{models_dir}/{model_name}/latest.pth')


def train_text_model(folder, *, models_dir, model_name, epochs=50, batch_size=8, lr=0.001, device='cuda'):
    os.makedirs(f'{models_dir}/{model_name}', exist_ok=True)

    # TODO: Save all of this in the .pth file with the model

    tree, label_set, files, labels = hierarchy_and_labels_from_folder(folder)
    with open(f'{models_dir}/{model_name}/tree.csv', 'w') as f:
        f.write('\n'.join(f'{parent},{child}' for parent, child in tree.edges()))
    with open(f'{models_dir}/{model_name}/labels.txt', 'w') as f:
        f.write('\n'.join(label_set))
    with open(f'{models_dir}/{model_name}/train.csv', 'w') as f:
        f.write('\n'.join(f'{file},{label}' for file, label in zip(files, labels)))
    with open(f'{models_dir}/{model_name}/meta.json', 'w') as f:
        json.dump({'type': 'text'}, f)

    transform = ROBERTA_BASE_ENCODER.transform()

    dataset = ClassiaTextDataset(files, labels, transform=transform)

    model = ClassiaTextModelV1(tree)
    model.to(device)
    loss = MarginLoss(tree, with_leaf_targets=False)

    optimizer = SGD(model.parameters(), lr=lr)

    def pad_tokens(batch):

        max_tokens = max((seq.shape[0] for seq, _, _ in batch))

        input_ids = torch.full((len(batch), max_tokens), fill_value=ROBERTA_BASE_ENCODER.encoderConf.padding_idx)
        for i in range(len(batch)):
            input_ids[i, :len(batch[i][0])] = batch[i][0]

        labels = torch.tensor([label for _, _, label in batch], dtype=torch.long)

        return input_ids, labels

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_tokens)

    model.train()

    for epoch in tqdm(range(epochs), desc="Training", position=0):
        for x, y in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}", position=1):
            x = x.to(device)

            optimizer.zero_grad()

            theta = model(x).cpu()
            loss_value = loss(theta, y)
            loss_value.backward()

            optimizer.step()

        torch.save(model.state_dict(), f'{models_dir}/{model_name}/latest.pth')
