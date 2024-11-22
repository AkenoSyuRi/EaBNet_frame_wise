import torch

from network.EaBNet import U2Net_Encoder
from network.EaBNet_FrameWise_Stateful import U2NetEncoder as U2Net_Encoder_FrameWise


def load_state_dict_from1to2(dict1, dict2):
    """dict1 and dict2 have the same value shapes and order."""
    assert len(dict1) == len(dict2)
    for (k1, v1), (k2, v2) in zip(dict1.items(), dict2.items()):
        assert v1.shape == v2.shape
        v2.copy_(v1)
    ...


def main():
    M, k1, k2, c, intra_connect, norm_type = 32, (2, 3), (1, 3), 64, "cat", "iLN"

    net1 = U2Net_Encoder(M * 2, k1, k2, c, intra_connect, norm_type)
    net2 = U2Net_Encoder_FrameWise(M * 2, k1, k2, c, intra_connect, norm_type)
    load_state_dict_from1to2(net1.state_dict(), net2.state_dict())
    net1.eval()
    net2.eval()

    x = torch.randn(1, 2 * M, 200, 161)
    states = [
        torch.zeros(1, 2 * M, 1, 161),
        torch.zeros(1, c, 1, 79),
        torch.zeros(1, c, 1, 39),
        torch.zeros(1, c, 1, 19),
        torch.zeros(1, c, 1, 9),
    ]

    y2_list = []
    with torch.no_grad():
        y1, _ = net1(x)
        for i in range(x.shape[2]):
            out2, _, states = net2(x[:, :, [i]], states)
            y2_list += [out2]
    y2 = torch.cat(y2_list, dim=2)

    torch.testing.assert_close(y1, y2)
    ...


if __name__ == "__main__":
    main()
    ...
