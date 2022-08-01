import torch.nn as nn
import torch.nn.functional as F
import torch

from .Config import *
from .AModel import *


class ImageEncoder(nn.Module):

    def __init__(self):
        super(ImageEncoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, (3, 3))
        self.conv2 = nn.Conv2d(32, 32, (3, 3))
        self.conv3 = nn.Conv2d(32, 64, (3, 3))
        self.conv4 = nn.Conv2d(64, 64, (3, 3))
        self.conv5 = nn.Conv2d(64, 128, (3, 3))
        self.conv6 = nn.Conv2d(128, 128, (3, 3))
        self.fc1 = nn.Linear(in_features=128 * 28 * 28, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=1024)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.dropout3 = nn.Dropout(0.25)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout3(x)

        x = x.view(-1, 128 * 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


class ContextEncoder(nn.Module):

    def __init__(self):
        super(ContextEncoder, self).__init__()

    def forward(self, x):
        pass


class Decoder(nn.Module):

    def __init__(self):
        super(Decoder, self).__init__()

    def forward(self, x):
        pass


class MyPix2code(nn.Module):

    def __init__(self):
        super(MyPix2code, self).__init__()
        self.image_encoder = ImageEncoder()
        self.context_encoder = ContextEncoder()
        self.decoder = Decoder()

    def forward(self, image, context):
        pass

