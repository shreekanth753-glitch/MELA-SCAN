import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """
    A convolutional building block consisting of:
    Conv2D -> BatchNorm -> ReLU -> MaxPool
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=True
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.block(x)


class MelaScanCNN(nn.Module):
    """
    Custom CNN for binary classification:

    0 = Healthy
    1 = Eczema

    The network is intentionally designed to have
    approximately 10 million trainable parameters.
    """

    def __init__(self, num_classes=2):
        super().__init__()

        self.features = nn.Sequential(
            ConvBlock(3, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
            ConvBlock(256, 256),
        )

        # The input images are 160x160.
        #
        # Five 2x2 pooling operations reduce:
        #
        # 160 -> 80 -> 40 -> 20 -> 10 -> 5
        #
        # However, we deliberately use adaptive pooling
        # so that the classifier receives a fixed 4x4 feature map.
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        self.classifier = nn.Sequential(
            nn.Flatten(),

            nn.Linear(256 * 4 * 4, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.40),

            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.30),

            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = self.classifier(x)
        return x


def create_model(num_classes=2):
    """
    Create a new MELA-SCAN CNN.
    """
    return MelaScanCNN(num_classes=num_classes)


def count_parameters(model):
    """
    Count trainable parameters.
    """
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


if __name__ == "__main__":
    model = create_model()

    print("=" * 60)
    print("MELA-SCAN MODEL")
    print("=" * 60)

    print(model)

    parameter_count = count_parameters(model)

    print("\nTrainable parameters:")
    print(f"{parameter_count:,}")

    dummy_input = torch.randn(1, 3, 160, 160)

    with torch.no_grad():
        output = model(dummy_input)

    print("\nInput shape:")
    print(tuple(dummy_input.shape))

    print("\nOutput shape:")
    print(tuple(output.shape))

    print("\nExpected output:")
    print("(batch_size, 2)")

    print("=" * 60)