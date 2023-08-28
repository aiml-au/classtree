from torch.nn import Module, Linear
from torch.nn.functional import pad
from torchtext.models import RobertaClassificationHead
from torchvision.models import resnet50
from torchvision.models.resnet import ResNet50_Weights
from torchtext.models.roberta.bundler import ROBERTA_BASE_ENCODER

from .hier import Hierarchy


class ClassiaImageModelV1(Module):

    def __init__(self, tree: Hierarchy):
        super().__init__()

        self.resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        hidden_units = self.resnet.fc.in_features
        self.resnet.fc = Linear(hidden_units, tree.num_nodes())

    def forward(self, x):
        return self.resnet(x)


class ClassiaTextModelV1(Module):

    def __init__(self, tree: Hierarchy):
        super().__init__()

        hidden_units = ROBERTA_BASE_ENCODER.encoderConf.embedding_dim

        self.classifier = RobertaClassificationHead(input_dim=hidden_units, num_classes=tree.num_nodes())
        self.roberta = ROBERTA_BASE_ENCODER.get_model(head=self.classifier)

    def forward(self, input_ids):
        return self.roberta(input_ids)
