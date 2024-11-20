import torch
from torch import nn

from network.EaBNet import SqueezedTCNGroup
from network.EaBNet_FrameWise import SqueezedTCNGroup as SqueezedTCNGroup_FrameWise


def main():
    x = torch.randn(3, 256, 200)
    q = 3

    net1 = nn.ModuleList([SqueezedTCNGroup(5, 64, 256, 6, True, "iLN") for _ in range(q)])
    net2 = nn.ModuleList([SqueezedTCNGroup_FrameWise(5, 64, 256, 6, True, "iLN") for _ in range(q)])

    net2.load_state_dict(net1.state_dict())
    net2 = torch.jit.script(net2)
    net1.eval()
    net2.eval()

    with torch.no_grad():
        y1 = x
        for j in range(q):
            y1 = net1[j](y1)

        y2 = x
        for j in range(q):
            y2_list = []
            for i in range(y2.shape[-1]):
                out2 = net2[j](y2[:, :, [i]])
                y2_list += [out2]
            y2 = torch.cat(y2_list, dim=-1)

    torch.testing.assert_close(y1, y2)
    ...


if __name__ == "__main__":
    main()
    ...
