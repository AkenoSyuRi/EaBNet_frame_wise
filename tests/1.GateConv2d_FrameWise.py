import torch

from network.EaBNet import GateConv2d
from network.EaBNet_FrameWise_Stateful import GateConv2dFW


def main():
    inputs = torch.randn(1, 16, 7, 161)
    states = torch.zeros(1, 16, 1, 161)

    net1 = GateConv2d(16, 32, (2, 3), (1, 3))
    net2 = GateConv2dFW(16, 32, (2, 3), (1, 3))
    net2.load_state_dict(net1.state_dict())
    net2 = torch.jit.script(net2)
    net1.eval()
    net2.eval()

    y2_list = []
    with torch.no_grad():
        y1 = net1(inputs)
        for i in range(inputs.shape[2]):
            out, states = net2(inputs[:, :, [i]], states)
            y2_list += [out]
    y2 = torch.cat(y2_list, dim=2)
    torch.testing.assert_close(y1, y2)
    ...


if __name__ == "__main__":
    main()
    ...
