import torch

from network.EaBNet import LSTM_BF
from network.EaBNet_FrameWise_Stateful import LSTM_BF_FW


def load_state_dict_from1to2(dict1, dict2):
    """dict1 and dict2 have the same value shapes and order."""
    assert len(dict1) == len(dict2)
    for (k1, v1), (k2, v2) in zip(dict1.items(), dict2.items()):
        assert v1.shape == v2.shape
        v2.copy_(v1)
    ...


def main():
    net1 = LSTM_BF(64, 8)
    net2 = LSTM_BF_FW(64, 8)
    load_state_dict_from1to2(net1.state_dict(), net2.state_dict())
    net2 = torch.jit.script(net2)
    net1.eval()
    net2.eval()

    x = torch.randn(1, 64, 200, 161)
    state = torch.zeros(2, 161, 64, 2)

    y2_list = []
    with torch.no_grad():
        y1 = net1(x)
        for i in range(x.shape[2]):
            out, state = net2(x[:, :, [i]], state)
            y2_list += [out]
    y2 = torch.cat(y2_list, dim=1)
    torch.testing.assert_close(y1, y2)
    ...


if __name__ == "__main__":
    main()
    ...
