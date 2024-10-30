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
        conv_types=None,
        pooling_types=None,
        activation="silu",
        use_gap=True,
    ):
        super(Encoder1D, self).__init__()

        # Activation function selection
        activation_fn = {
            "relu": nn.ReLU(),
            "silu": nn.SiLU(),
            "gelu": nn.GELU(),
            "leaky_relu": nn.LeakyReLU(),
        }[activation]

        # Validate input lengths
        if pooling_types is None:
            pooling_types = ["max"] * len(hidden_channels)
        if conv_types is None:
            conv_types = ["standard"] * len(hidden_channels)

        if len(pooling_types) != len(hidden_channels) or len(conv_types) != len(
            hidden_channels
        ):
            raise ValueError(
                "Length of pooling_types and conv_types must match hidden_channels."
            )

        layers = []
        current_channels = in_channels

        # Build layers
        for i in range(len(hidden_channels)):
            out_channels = hidden_channels[i]
            conv_type = conv_types[i]

            # Select the convolution type
            if conv_type == "standard":
                conv = nn.Conv1d(
                    current_channels,
                    out_channels,
                    kernel_size,
                    padding=kernel_size // 2,
                    stride=stride,
                    bias=False,
                )
            elif conv_type == "depthwise":
                conv = nn.Conv1d(
                    current_channels,
                    out_channels,
                    kernel_size,
                    padding=kernel_size // 2,
                    stride=stride,
                    groups=current_channels,
                    bias=False,
                )
            elif conv_type == "separable":
                conv = nn.Sequential(
                    nn.Conv1d(
                        current_channels,
                        current_channels,
                        kernel_size,
                        padding=kernel_size // 2,
                        stride=stride,
                        groups=current_channels,
                        bias=False,
                    ),
                    nn.Conv1d(current_channels, out_channels, 1, bias=False),
                )
            else:
                raise ValueError(f"Unsupported conv_type: {conv_type}")

            layers.extend([conv, nn.BatchNorm1d(out_channels), activation_fn])

            # Apply pooling
            pooling_type = pooling_types[i]
            if pooling_type == "max":
                layers.append(nn.MaxPool1d(kernel_size=3, stride=2, padding=1))
            elif pooling_type == "avg":
                layers.append(nn.AvgPool1d(kernel_size=3, stride=2, padding=1))
            elif pooling_type == "both":
                layers.append(ParallelPool(kernel_size=3, stride=2, padding=1))
                out_channels *= 2  # Double the channels when using ParallelPool
            elif pooling_type is None or pooling_type == "none":
                pass
            else:
                raise ValueError(f"Unsupported pooling_type: {pooling_type}")

            current_channels = out_channels

        self.features = nn.Sequential(*layers)

        # Classification head
        self.use_gap = use_gap
        if use_gap:
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(current_channels, num_classes),
            )
        else:
            self.classifier = nn.Linear(current_channels, num_classes)

    def forward(self, x):
        x = self.features(x)
        if self.use_gap:
            x = self.classifier(x)
        else:
            x = x.mean(dim=-1)
            x = self.classifier(x)
        return x


class ParallelPool(nn.Module):
    """Parallel pooling layer to combine max and avg pooling."""

    def __init__(self, kernel_size, stride, padding):
        super(ParallelPool, self).__init__()
        self.max_pool = nn.MaxPool1d(kernel_size, stride, padding)
        self.avg_pool = nn.AvgPool1d(kernel_size, stride, padding)

    def forward(self, x):
        max_pooled = self.max_pool(x)
        avg_pooled = self.avg_pool(x)
        # Concatenate along the channel dimension
        return torch.cat((max_pooled, avg_pooled), dim=1)
