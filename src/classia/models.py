from torch.nn import Module, Linear
from torch.nn.functional import pad
from torchtext.models import RobertaClassificationHead
from torchvision.models import resnet18, resnet50, resnet101
from torchvision.models.resnet import ResNet18_Weights, ResNet50_Weights, ResNet101_Weights
from torchtext.models.roberta.bundler import ROBERTA_BASE_ENCODER, ROBERTA_LARGE_ENCODER

from .hier import Hierarchy
import logging
LOGGER = logging.getLogger(__name__)

# Image model: Small, Medium, Large
# ref: https://pytorch.org/vision/master/_modules/torchvision/models/resnet.html
class ClassiaImageModelSmall(Module):

    def __init__(self, tree: Hierarchy):
        super().__init__()

        self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT) # 183250
        hidden_units = self.resnet.fc.in_features
        self.resnet.fc = Linear(hidden_units, tree.num_nodes())

    def forward(self, x):
        return self.resnet(x)

class ClassiaImageModelMedium(Module):

    def __init__(self, tree: Hierarchy):
        super().__init__()

        self.resnet = resnet50(weights=ResNet50_Weights.DEFAULT) # 183250
        hidden_units = self.resnet.fc.in_features
        self.resnet.fc = Linear(hidden_units, tree.num_nodes())

    def forward(self, x):
        return self.resnet(x)
    
class ClassiaImageModelLarge(Module):

    def __init__(self, tree: Hierarchy):
        super().__init__()

        self.resnet = resnet101(weights=ResNet101_Weights.DEFAULT) # 183250
        hidden_units = self.resnet.fc.in_features
        self.resnet.fc = Linear(hidden_units, tree.num_nodes())

    def forward(self, x):
        return self.resnet(x)


# TEXT MODEL: Base, Large
# ref: https://pytorch.org/text/stable/_modules/torchtext/models/roberta/bundler.html
class ClassiaTextModelBase(Module):

    def __init__(self, tree: Hierarchy):
        super().__init__()

        hidden_units = ROBERTA_BASE_ENCODER.encoderConf.embedding_dim

        self.classifier = RobertaClassificationHead(input_dim=hidden_units, num_classes=tree.num_nodes())
        self.roberta = ROBERTA_BASE_ENCODER.get_model(head=self.classifier)

    def forward(self, input_ids):
        return self.roberta(input_ids)
    
class ClassiaTextModelLarge(Module):

    def __init__(self, tree: Hierarchy):
        super().__init__()

        hidden_units = ROBERTA_LARGE_ENCODER.encoderConf.embedding_dim

        self.classifier = RobertaClassificationHead(input_dim=hidden_units, num_classes=tree.num_nodes())
        self.roberta = ROBERTA_LARGE_ENCODER.get_model(head=self.classifier)

    def forward(self, input_ids):
        return self.roberta(input_ids)
    

    
def get_image_model(tree, image_model_size):
    if image_model_size == 'small':
        model = ClassiaImageModelSmall(tree)
    elif image_model_size == 'medium':
        model = ClassiaImageModelMedium(tree)
    elif image_model_size == 'large':
        model = ClassiaImageModelLarge(tree)

    return model


def get_text_model(tree, text_model_size):
    if text_model_size == 'base':
        model = ClassiaTextModelBase(tree)
    elif text_model_size == 'large':
        model = ClassiaTextModelLarge(tree)
        
    return model


def get_text_encoder(text_model_size):
    if text_model_size == 'base':
        encoder = ROBERTA_BASE_ENCODER
    elif text_model_size == 'large':
        encoder = ROBERTA_LARGE_ENCODER
        
    return encoder
