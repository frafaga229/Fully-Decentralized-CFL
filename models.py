import torch.nn as nn


class LinearModel(nn.Module):
    def __init__(self, input_dimension, output_dimension):
        super(LinearModel, self).__init__()
        self.input_dimension = input_dimension
        self.num_classes = output_dimension
        self.fc = nn.Linear(input_dimension, output_dimension)

    def forward(self, x):
        return self.fc(x)
