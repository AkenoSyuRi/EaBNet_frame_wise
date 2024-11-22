import torch

from network.EaBNet import U2Net_Decoder
from network.EaBNet_FrameWise_Stateful import U2NetDecoder as U2Net_Decoder_FrameWise


def load_state_dict_from1to2(dict1, dict2):
    """dict1 and dict2 have the same value shapes and order."""
    assert len(dict1) == len(dict2)
    for (k1, v1), (k2, v2) in zip(dict1.items(), dict2.items()):
        assert v1.shape == v2.shape
        v2.copy_(v1)
    ...


def main():
    net1 = U2Net_Decoder(64, 64, (2, 3), (1, 3), "cat", "iLN")
    net2 = U2Net_Decoder_FrameWise(64, 64, (2, 3), (1, 3), "cat", "iLN")
    load_state_dict_from1to2(net1.state_dict(), net2.state_dict())
    net1.eval()
    net2.eval()

    x = torch.randn(1, 64, 200, 4)
    en_list = [
        torch.randn(1, 64, 200, 79),
        torch.randn(1, 64, 200, 39),
        torch.randn(1, 64, 200, 19),
        torch.randn(1, 64, 200, 9),
        torch.randn(1, 64, 200, 4),
    ]
    states = [
        torch.zeros(1, 128, 1, 4),
        torch.zeros(1, 128, 1, 9),
        torch.zeros(1, 128, 1, 19),
        torch.zeros(1, 128, 1, 39),
        torch.zeros(1, 128, 1, 79),
    ]

    y2_list = []
    with torch.no_grad():
        y1 = net1(x, en_list)
        for i in range(x.shape[2]):
            en_fw_list = list(map(lambda x: x[:, :, [i]], en_list))
            out, states = net2(x[:, :, [i]], en_fw_list, states)
            y2_list += [out]
    y2 = torch.cat(y2_list, dim=2)
    torch.testing.assert_close(y1, y2)
    ...


if __name__ == "__main__":
    main()
    ...
