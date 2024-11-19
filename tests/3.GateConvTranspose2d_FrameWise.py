from typing import Tuple

import torch
from torch import nn, Tensor

from network.EaBNet import GateConvTranspose2d


class GateConvTranspose2dFW(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int],
        stride: Tuple[int, int],
    ):
        super(GateConvTranspose2dFW, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        assert stride[0] == 1, f"{self.__class__.__name__} only supports stride[0] == 1"
        self.state = None

        k_t = kernel_size[0]
        if k_t > 1:
            self.conv = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=in_channels,
                    out_channels=out_channels * 2,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=(k_t - 1, 0),
                ),
                nn.Identity(),
            )
        else:
            self.conv = nn.ConvTranspose2d(
                in_channels=in_channels, out_channels=out_channels * 2, kernel_size=kernel_size, stride=stride
            )

    def forward(self, inputs: Tensor) -> Tensor:
        """inputs: (batch_size, channels, time=1, freq)"""
        assert inputs.shape[-2] == 1, f"{self.__class__.__name__} only supports single frame input"

        if inputs.ndim == 3:
            inputs = inputs.unsqueeze(dim=1)

        if self.state is None:
            B, C, _, F = inputs.shape
            state = torch.zeros(B, C, self.kernel_size[0] - 1, F, dtype=inputs.dtype, device=inputs.device)
            inputs = torch.cat([state, inputs], dim=2)
        else:
            inputs = torch.cat([self.state, inputs], dim=2)
        self.state = inputs[:, :, 1:]

        x = self.conv(inputs)
        outputs, gate = x.chunk(2, dim=1)
        return outputs * gate.sigmoid()


def main():
    x = torch.randn(3, 128, 200, 4)

    net1 = GateConvTranspose2d(128, 128, (2, 3), (1, 2))
    net2 = GateConvTranspose2dFW(128, 128, (2, 3), (1, 2))
    net2.load_state_dict(net1.state_dict())
    net1.eval()
    net2.eval()

    y2_list = []
    with torch.no_grad():
        y1 = net1(x)
        for i in range(x.shape[2]):
            y2_list += [net2(x[:, :, [i]])]
    y2 = torch.cat(y2_list, dim=2)
    torch.testing.assert_close(y1, y2)
    ...


if __name__ == "__main__":
    main()
    ...
