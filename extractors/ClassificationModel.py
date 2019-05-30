import torchvision.models as models
import torch.nn as nn
import torchvision.transforms as Transform

from PIL import Image
from .base import FeatureExtractor


class DetectionModelBasedExtractor(FeatureExtractor):
    type = 'detectionnetwork'

    def __init__(self, params):
        self.input_shape = params['input_shape']
        self.pre_process = self.preprocess()
        self.device = params['device']
        base_model_name = params['base_model']
        self.base_model = self.get_base_model(base_model_name)
        self.model = ModelBasedExtractor(self.base_model)
        self.model.eval()

    def preprocess(self)->Transform:
        # default vgg16 input: (224, 224)
        scale_image = Transform.Resize((224, 224))
        # vgg16 Imagenet normalization, source: https://github.com/pytorch/examples/blob/master/imagenet/main.py#L92-L93
        normalize = Transform.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])

        pre_process = Transform.Compose([scale_image,
                                         Transform.ToTensor(),
                                         normalize])
        return pre_process

    def get_base_model(self, base_model_name: str):
        if base_model_name == 'vgg16':
            model = models.vgg16(pretrained=True)
        if base_model_name == 'vgg16_bn':
            model = models.vgg16_bn(pretrained=True)
        else:
            raise Exception('Base model not supported: use vgg16 or vgg16_bn')

        model = model.to(self.device)
        for param in model.features.parameters():
            param.require_grad = False
        return model

    def extract_features(self, img: Image):
        img = self.pre_process(img).unsqueeze(0)
        img = img.to(self.device)
        feature_vector = self.model(img)
        return feature_vector.cpu().detach().numpy().flatten()


class ModelBasedExtractor(nn.Module):
    def __init__(self, original_model: nn.Module):
        super(ModelBasedExtractor, self).__init__()
        self.features = original_model.features
        self.avgpool = original_model.avgpool
        self.linear = nn.Sequential(*list(original_model.classifier.children())[:-1])

    def forward(self, input):
        cnn_features = self.features(input)
        pooled_features = self.avgpool(cnn_features)
        pooled_features = pooled_features.view(-1, 512 * 7 * 7)
        linear_features = self.linear(pooled_features)
        return linear_features
