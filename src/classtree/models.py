from torch.nn import Module, Linear
from torchtext.models import RobertaClassificationHead
from torchvision.models import resnet18, resnet50, resnet101
from torchvision.models.resnet import (
    ResNet18_Weights,
    ResNet50_Weights,
    ResNet101_Weights,
)
from torchtext.models.roberta.bundler import ROBERTA_BASE_ENCODER, ROBERTA_LARGE_ENCODER

from .hier import Hierarchy
import logging

LOGGER = logging.getLogger(__name__)


# Image model: Small, Medium, Large
# ref: https://pytorch.org/vision/master/_modules/torchvision/models/resnet.html
class ClasstreeImageModelV1Small(Module):
    def __init__(self, tree: Hierarchy):
        super().__init__()

        self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        hidden_units = self.resnet.fc.in_features
        self.resnet.fc = Linear(hidden_units, tree.num_nodes())

    def forward(self, x):
        return self.resnet(x)


class ClasstreeImageModelV1Medium(Module):
    def __init__(self, tree: Hierarchy):
        super().__init__()

        self.resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        hidden_units = self.resnet.fc.in_features
        self.resnet.fc = Linear(hidden_units, tree.num_nodes())

    def forward(self, x):
        return self.resnet(x)


class ClasstreeImageModelV1Large(Module):
    def __init__(self, tree: Hierarchy):
        super().__init__()

        self.resnet = resnet101(weights=ResNet101_Weights.DEFAULT)
        hidden_units = self.resnet.fc.in_features
        self.resnet.fc = Linear(hidden_units, tree.num_nodes())

    def forward(self, x):
        return self.resnet(x)


# TEXT MODEL: Base, Large
# ref: https://pytorch.org/text/stable/_modules/torchtext/models/roberta/bundler.html
class ClasstreeTextModelV1Medium(Module):
    def __init__(self, tree: Hierarchy):
        super().__init__()

        hidden_units = ROBERTA_BASE_ENCODER.encoderConf.embedding_dim

        self.classifier = RobertaClassificationHead(
            input_dim=hidden_units, num_classes=tree.num_nodes()
        )
        self.roberta = ROBERTA_BASE_ENCODER.get_model(head=self.classifier)

    def forward(self, input_ids):
        return self.roberta(input_ids)


class ClasstreeTextModelV1Large(Module):
    def __init__(self, tree: Hierarchy):
        super().__init__()

        hidden_units = ROBERTA_LARGE_ENCODER.encoderConf.embedding_dim

        self.classifier = RobertaClassificationHead(
            input_dim=hidden_units, num_classes=tree.num_nodes()
        )
        self.roberta = ROBERTA_LARGE_ENCODER.get_model(head=self.classifier)

    def forward(self, input_ids):
        return self.roberta(input_ids)


model_classes = {
    "image-v1-s": ClasstreeImageModelV1Small,
    "image-v1-m": ClasstreeImageModelV1Medium,
    "image-v1-l": ClasstreeImageModelV1Large,
    "text-v1-m": ClasstreeTextModelV1Medium,
    "text-v1-l": ClasstreeTextModelV1Large,
}


def get_model_id(type, version, size):
    if type not in ["image", "text"]:
        raise ValueError(f"Invalid model type: {type}")
    if version != 1:
        raise ValueError(f"Invalid model version: {version}")
    if size not in ["s", "m", "l"]:
        raise ValueError(f"Invalid model size: {size}")

    return f"{type}-v{version}-{size}"


def get_text_encoder(text_model_size):
    if text_model_size == "m":
        encoder = ROBERTA_BASE_ENCODER
    elif text_model_size == "l":
        encoder = ROBERTA_LARGE_ENCODER
    else:
        raise ValueError(f"Invalid text model size: {text_model_size}")

    return encoder
