import torch

from network.EaBNet import GateConvTranspose2d
from network.EaBNet_FrameWise_Stateful import GateConvTranspose2dFW


def main():
    inputs = torch.randn(1, 128, 200, 4)
    states = torch.zeros(1, 128, 1, 4)

    net1 = GateConvTranspose2d(128, 128, (2, 3), (1, 2))
    net2 = GateConvTranspose2dFW(128, 128, (2, 3), (1, 2))
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
