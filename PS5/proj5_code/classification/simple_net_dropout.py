import torch
import torch.nn as nn


class SimpleNetDropout(nn.Module):
    def __init__(self):
        """
        Init function to define the layers and loss function

        Note: Use 'sum' reduction in the loss_criterion. Read Pytorch documention to understand what it means
        """
        super().__init__()

        self.conv_layers = nn.Sequential()
        self.fc_layers = nn.Sequential()
        self.loss_criterion = nn.CrossEntropyLoss(reduction = 'sum')

        ############################################################################
        # Student code begin
        ############################################################################
        KSIZE = 3
        n1 = 30
        n2 = 30
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, n1, KSIZE),
            nn.BatchNorm2d(n1),
            nn.Dropout(.1),
            nn.ReLU(),
            nn.MaxPool2d(KSIZE),
            #nn.ReLU(),
            nn.Conv2d(n1, n2, KSIZE),
            nn.BatchNorm2d(n2),
            nn.ReLU(),
            nn.MaxPool2d(KSIZE)
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(1080, 150),
            nn.Dropout(.1),
            nn.ReLU(),
            nn.Linear(150, 15)
        )
        ############################################################################
        # Student code end
        ############################################################################

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Perform the forward pass with the net

        Note: do not perform soft-max or convert to probabilities in this function

        Args:
        -   x: the input image [Dim: (N,C,H,W)]
        Returns:
        -   y: the output (raw scores) of the net [Dim: (N,15)]
        """
        conv_features = None  # output of x passed through convolution layers (4D tensor)
        flattened_conv_features = None  # conv_features reshaped into 2D tensor using .reshape()
        model_output = None  # output of flattened_conv_features passed through fully connected layers
        ############################################################################
        # Student code begin
        ############################################################################
        conv_features = self.conv_layers(x)
        flat = nn.Flatten()
        flattened_conv_features = flat(conv_features)
        model_output = self.fc_layers(flattened_conv_features)
        ############################################################################
        # Student code end
        ############################################################################

        return model_output
