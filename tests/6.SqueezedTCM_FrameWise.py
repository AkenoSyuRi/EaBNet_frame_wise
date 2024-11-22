import torch

from network.EaBNet import SqueezedTCM
from network.EaBNet_FrameWise_Stateful import SqueezedTCM_FW


def load_state_dict_from1to2(dict1, dict2):
    """dict1 and dict2 have the same value shapes and order."""
    assert len(dict1) == len(dict2)
    for (k1, v1), (k2, v2) in zip(dict1.items(), dict2.items()):
        assert v1.shape == v2.shape
        v2.copy_(v1)
    ...


def main():
    dilation, kd1 = 2**3, 5
    net1 = SqueezedTCM(kd1, 64, dilation, 256, True, "iLN")
    net2 = SqueezedTCM_FW(kd1, 64, dilation, 256, True, "iLN")
    load_state_dict_from1to2(net1.state_dict(), net2.state_dict())
    net1.eval()
    net2.eval()

    x = torch.randn(1, 256, 200)
    state = torch.zeros(1, 64, (kd1 - 1) * dilation, 2)

    y2_list = []
    with torch.no_grad():
        y1 = net1(x)
        for i in range(x.shape[-1]):
            out2, state = net2(x[:, :, [i]], state)
            y2_list += [out2]
    y2 = torch.cat(y2_list, dim=-1)
    torch.testing.assert_close(y1, y2)
    ...


if __name__ == "__main__":
    main()
    ...
