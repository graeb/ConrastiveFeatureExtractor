import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812


class PDNModel(nn.Module):
    """Patch Description Network small.

    Args:
        out_channels (int): number of convolution output channels
        padding (bool): use padding in convoluional layers
            Defaults to ``False``.
    """

    def __init__(
        self, out_channels: int, padding: bool = False, disable_batch_norm: bool = False
    ) -> None:
        super().__init__()
        pad_mult = 1 if padding else 0
        self.disable_batch_norm = disable_batch_norm
        self.conv1 = nn.Conv2d(3, 128, kernel_size=4, stride=1, padding=3 * pad_mult)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=3 * pad_mult)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1 * pad_mult)
        self.conv4 = nn.Conv2d(
            256, out_channels, kernel_size=4, stride=1, padding=0 * pad_mult
        )
        self.avgpool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult)
        self.avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult)

    def forward(self, x: torch.Tensor):
        """Perform a forward pass through the network.

        Args:
            x (torch.Tensor): Input batch.

        Returns:
            dict: Dictionary containing intermediate results before and after ReLU.
        """
        results = {}

        # if not self.disable_batch_norm:
        #     x = imagenet_norm_batch(x)

        # Step 1: Conv1
        x1_pre_relu = self.conv1(x)
        x1 = F.relu(x1_pre_relu)
        results["layer1"] = x1_pre_relu
        results["layer1_act"] = x1

        # Step 2: AvgPool1
        x1_avg = self.avgpool1(x1)
        results["layer1_avg"] = x1_avg

        # Step 3: Conv2
        x2_pre_relu = self.conv2(x1_avg)
        x2 = F.relu(x2_pre_relu)
        results["layer2"] = x2_pre_relu
        results["layer2_act"] = x2

        # Step 4: AvgPool2
        x2_avg = self.avgpool2(x2)
        results["layer2_avg"] = x2_avg

        # Step 5: Conv3
        x3_pre_relu = self.conv3(x2_avg)
        # x3 = F.relu(x3_pre_relu)
        x3 = F.leaky_relu(x3_pre_relu)
        results["layer3"] = x3_pre_relu
        results["layer3_act"] = x3

        # Step 6: Conv4 (final layer, no ReLU)
        x4 = self.conv4(x3)
        results["layer4"] = x4

        return results
