import torch
from torch import nn

from network.EaBNet import SqueezedTCNGroup
from network.EaBNet_FrameWise_Stateful import SqueezedTCNGroup as SqueezedTCNGroup_FrameWise


def load_state_dict_from1to2(dict1, dict2):
    """dict1 and dict2 have the same value shapes and order."""
    assert len(dict1) == len(dict2)
    for (k1, v1), (k2, v2) in zip(dict1.items(), dict2.items()):
        assert v1.shape == v2.shape
        v2.copy_(v1)
    ...


def main():
    kd1, q, p = 5, 3, 6

    net1 = nn.ModuleList([SqueezedTCNGroup(kd1, 64, 256, p, True, "iLN") for _ in range(q)])
    net2 = nn.ModuleList([SqueezedTCNGroup_FrameWise(kd1, 64, 256, p, True, "iLN") for _ in range(q)])
    load_state_dict_from1to2(net1.state_dict(), net2.state_dict())
    net2 = torch.jit.script(net2)
    net1.eval()
    net2.eval()

    x = torch.randn(1, 256, 200)
    states = [[torch.zeros(1, 64, (kd1 - 1) * 2**i, 2) for i in range(p)] for _ in range(q)]

    with torch.no_grad():
        y1 = x
        for j in range(q):
            y1 = net1[j](y1)

        y2 = x
        for j in range(q):
            y2_list = []
            for i in range(y2.shape[-1]):
                out2, states[j] = net2[j](y2[:, :, [i]], states[j])
                y2_list += [out2]
            y2 = torch.cat(y2_list, dim=-1)

    torch.testing.assert_close(y1, y2)
    ...


if __name__ == "__main__":
    main()
    ...
