import torchvision
import torch.nn as nn

class DenseNet121(nn.Module):
    """Model modified.
    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.
    """
    def __init__(self, out_size):
        super(DenseNet121, self).__init__()
        self.densenet121 = torchvision.models.densenet121(pretrained=True)
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Linear(num_ftrs, out_size)

    def forward(self, x):
        x = self.densenet121(x)
        return x

class ResNet152(nn.Module):
    """Model modified.
    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.
    """
    def __init__(self, out_size):
        super(ResNet152, self).__init__()
        self.resnet152 = torchvision.models.resnet152(pretrained=True)
        num_ftrs = self.resnet152.fc.in_features
        self.resnet152.fc = nn.Linear(num_ftrs, out_size)

    def forward(self, x):
        x = self.resnet152(x)
        return x

class ResNet18(nn.Module):
    """Model modified.
    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.
    """
    def __init__(self, out_size):
        super(ResNet18, self).__init__()
        self.resnet18 = torchvision.models.resnet18(pretrained=True)
        num_ftrs = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Linear(num_ftrs, out_size)

    def forward(self, x):
        x = self.resnet18(x)
        return x

class GoogleNet(nn.Module):
    """Model modified.
    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.
    """
    def __init__(self, out_size):
        super(GoogleNet, self).__init__()
        self.googlenet = torchvision.models.googlenet(pretrained=True)
        num_ftrs = self.googlenet.fc.in_features
        self.googlenet.fc = nn.Linear(num_ftrs, out_size)

    def forward(self, x):
        x = self.googlenet(x)
        return x