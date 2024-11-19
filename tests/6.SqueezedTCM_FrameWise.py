import torch
from torch import nn, Tensor

from network.EaBNet import SqueezedTCM, NormSwitch


class CausalConstantPad1d(nn.Module):
    def __init__(self, padding, value=0.0):
        super(CausalConstantPad1d, self).__init__()
        self.padding = padding
        self.value = value
        self.state = None

    def forward(self, inputs: Tensor):
        assert inputs.shape[-1] == 1, "CausalConstantPad1d only supports 1 time frame input"

        if self.state is None:
            inputs = nn.functional.pad(inputs, (*self.padding, 0, 0), "constant", self.value)
        else:
            inputs = torch.cat([self.state, inputs], dim=-1)
        self.state = inputs[..., 1:]
        return inputs


class SqueezedTCM_FW(nn.Module):
    def __init__(
        self,
        kd1: int,
        cd1: int,
        dilation: int,
        d_feat: int,
        is_causal: bool,
        norm_type: str,
    ):
        super(SqueezedTCM_FW, self).__init__()
        self.kd1 = kd1
        self.cd1 = cd1
        self.dilation = dilation
        self.d_feat = d_feat
        self.is_causal = is_causal
        self.norm_type = norm_type
        self.state1 = None
        self.state2 = None

        self.in_conv = nn.Conv1d(d_feat, cd1, 1, bias=False)
        if is_causal:
            self.pad = ((kd1 - 1) * dilation, 0)
        else:
            self.pad = ((kd1 - 1) * dilation // 2, (kd1 - 1) * dilation // 2)
        self.left_conv = nn.Sequential(
            nn.PReLU(cd1),
            NormSwitch(norm_type, "1D", cd1),
            CausalConstantPad1d(self.pad),
            nn.Conv1d(cd1, cd1, kd1, dilation=dilation, bias=False),
        )
        self.right_conv = nn.Sequential(
            nn.PReLU(cd1),
            NormSwitch(norm_type, "1D", cd1),
            CausalConstantPad1d(self.pad),
            nn.Conv1d(cd1, cd1, kernel_size=kd1, dilation=dilation, bias=False),
            nn.Sigmoid(),
        )
        self.out_conv = nn.Sequential(
            nn.PReLU(cd1), NormSwitch(norm_type, "1D", cd1), nn.Conv1d(cd1, d_feat, kernel_size=1, bias=False)
        )

    def forward(self, x):
        resi = x
        x = self.in_conv(x)
        x = self.left_conv(x) * self.right_conv(x)
        x = self.out_conv(x)
        x = x + resi
        return x


def main():
    x = torch.randn(3, 256, 200)
    dilation = 2**3
    net1 = SqueezedTCM(5, 64, dilation, 256, True, "iLN")
    net2 = SqueezedTCM_FW(5, 64, dilation, 256, True, "iLN")
    net2.load_state_dict(net1.state_dict())
    net1.eval()
    net2.eval()

    y2_list = []
    with torch.no_grad():
        y1 = net1(x)
        for i in range(x.shape[-1]):
            out2 = net2(x[:, :, [i]])
            y2_list += [out2]
    y2 = torch.cat(y2_list, dim=-1)
    torch.testing.assert_close(y1, y2)
    ...


if __name__ == "__main__":
    main()
    ...
