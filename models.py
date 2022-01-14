import torch.nn as nn


class LinearModel(nn.Module):
    def __init__(self, input_dimension, num_classes):
        super(LinearModel, self).__init__()
        self.input_dimension = input_dimension
        self.num_classes = num_classes
        self.fc = nn.Linear(input_dimension, num_classes)

    def forward(self, x):
        return self.fc(x)
