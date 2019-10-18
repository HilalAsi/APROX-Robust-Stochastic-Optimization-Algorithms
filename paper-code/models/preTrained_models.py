import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

import torchvision

class FineTuneModel(nn.Module):
    def __init__(self, original_model, arch, num_classes, num_training_layers=1):
        super(FineTuneModel, self).__init__()

        if arch.startswith('alexnet'):
            self.features = original_model.features
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(256 * 6 * 6, 4096),
                nn.ELU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ELU(inplace=True),
                nn.Linear(4096, num_classes),
            )
            self.modelName = 'alexnet'
        elif arch.startswith('resnet'):
            # Everything except the last linear layer
            self.features = nn.Sequential(*list(original_model.children())[:-1])
            self.classifier = nn.Sequential(
                nn.Linear(512, num_classes)
            )
            self.modelName = 'resnet'
        elif arch.startswith('vgg16'):
            self.features = original_model.features
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(25088, 4096),
                nn.ELU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ELU(inplace=True),
                nn.Linear(4096, num_classes),
            )
            self.modelName = 'vgg16'
        else :
            raise("Finetuning not supported on this architecture yet")



        if self.modelName == 'vgg16':
            z = 1
            #for p in self.features.parameters():
                    #p.requires_grad = False
        else:
            # Freeze those weights
            num_layers = 0
            for p in self.features.parameters():
                num_layers += 1

            curr_layer = 0
            for p in self.features.parameters():
                if curr_layer+num_training_layers-1 < num_layers:
                #if num_training_layers=1 then train only the classifier
                    p.requires_grad = False
                curr_layer += 1

    def forward(self, x):
        f = self.features(x)
        if self.modelName == 'alexnet' :
            f = f.view(f.size(0), 256 * 6 * 6)
        elif self.modelName == 'vgg16':
            f = f.view(f.size(0), -1)
        elif self.modelName == 'resnet' :
            f = f.view(f.size(0), -1)
        y = self.classifier(f)
        return y, f

# num_training_layers - the last num_training_layers will be trained.
def pretrained_ResNet18_ELU(in_channels=3, out_channels=10, num_training_layers=1):
    original_model = torchvision.models.resnet18(pretrained=True)  # TODO change relu to elu?
    return FineTuneModel(original_model, 'resnet', out_channels, num_training_layers)

def pretrained_AlexNet_ELU(in_channels=3, out_channels=10, num_training_layers=1):
    original_model = torchvision.models.alexnet(pretrained=True)  # TODO change relu to elu?
    return FineTuneModel(original_model, 'alexnet', out_channels, num_training_layers)

def pretrained_vgg16_ELU(in_channels=3, out_channels=10,  num_training_layers=1):
    original_model = torchvision.models.vgg16(pretrained=True)  # TODO change relu to elu?
    return FineTuneModel(original_model, 'vgg16', out_channels, num_training_layers)

