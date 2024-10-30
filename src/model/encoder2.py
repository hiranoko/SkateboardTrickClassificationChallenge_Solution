import torch
import torch.nn as nn


class Encoder1D(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        kernel_size,
        num_classes=3,
        stride=1,
        conv_type="standard",  # 'standard', 'depthwise', or 'separable'
        pooling_type="max",  # 'max', 'avg', or 'both'
        activation="silu",  # 'relu', 'silu', etc.
        num_layers=1,  # Number of convolutional blocks
        use_gap=True,  # Use Global Average Pooling
    ):
        super(Encoder1D, self).__init__()

        # Select activation function
        activation_fn = {
            "relu": nn.ReLU(),
            "silu": nn.SiLU(),
            "gelu": nn.GELU(),
            "leaky_relu": nn.LeakyReLU(),
        }[activation]

        # Initialize layers list
        layers = []

        # First convolutional layer
        conv_layer = nn.Conv1d(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )
        layers.extend([conv_layer, nn.BatchNorm1d(hidden_channels), activation_fn])
        # Add additional convolutional layers
        for _ in range(num_layers - 1):
            if conv_type == "standard":
                conv = nn.Conv1d(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    stride=stride,
                    bias=False,
                )
            elif conv_type == "depthwise":
                conv = nn.Conv1d(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    stride=stride,
                    groups=hidden_channels,
                    bias=False,
                )
            elif conv_type == "separable":
                conv = nn.Sequential(
                    nn.Conv1d(
                        in_channels=hidden_channels,
                        out_channels=hidden_channels,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2,
                        stride=stride,
                        groups=hidden_channels,
                        bias=False,
                    ),
                    nn.Conv1d(
                        in_channels=hidden_channels,
                        out_channels=hidden_channels,
                        kernel_size=1,
                        bias=False,
                    ),
                )
            else:
                raise ValueError(f"Unsupported conv_type: {conv_type}")

            layers.extend([conv, nn.BatchNorm1d(hidden_channels), activation_fn])

            # Add pooling layer
            if pooling_type == "max":
                layers.append(nn.MaxPool1d(kernel_size=3, stride=2, padding=1))
            elif pooling_type == "avg":
                layers.append(nn.AvgPool1d(kernel_size=3, stride=2, padding=1))
            elif pooling_type == "both":
                layers.append(
                    nn.Sequential(
                        nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
                        nn.AvgPool1d(kernel_size=3, stride=2, padding=1),
                    )
                )
            else:
                raise ValueError(f"Unsupported pooling_type: {pooling_type}")

        self.features = nn.Sequential(*layers)

        # Classification head
        self.use_gap = use_gap
        if use_gap:
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(hidden_channels, num_classes),
            )
        else:
            self.classifier = nn.Linear(hidden_channels, num_classes)

    def forward(self, x):
        x = self.features(x)
        if self.use_gap:
            x = self.classifier(x)
        else:
            x = x.mean(dim=-1)  # Alternatively, you can use any other pooling
            x = self.classifier(x)
        return x
