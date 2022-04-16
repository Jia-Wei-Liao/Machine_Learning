import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b4


class EfficientNet_b4(nn.Module):
    def __init__(self, num_classes=219):
        super(EfficientNet_b4, self).__init__()
        model = efficientnet_b4(pretrained=True)
        model.classifier[1] = torch.nn.Linear(
            model.classifier[1].in_features, num_classes)
        model_child = list(model.children())
        self.feature_extr = nn.Sequential(*model_child[:-2])
        self.avgpool = model_child[-2]
        self.classifier = model_child[-1]
        self.upsample = nn.Upsample(size=(224, 224))

    def save(self, path):
        torch.save(self, path)

        return

    def forward(self, inputs, use_mc_loss=False, *args, **kwargs):
        x = self.upsample(inputs)
        feature = self.feature_extr(x)
        x = self.avgpool(feature)
        x = x.view(x.size(0), -1)
        outputs = self.classifier(x)

        if use_mc_loss:
            return outputs, feature

        else:
            return outputs

if __name__ == '__main__':
    model = EfficientNet_b4(100)
    #print(model)
