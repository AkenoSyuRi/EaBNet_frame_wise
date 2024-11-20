import torch

from network.EaBNet import U2Net_Decoder
from network.EaBNet_FrameWise import U2Net_Decoder as U2Net_Decoder_FrameWise


def main():
    x = torch.randn(3, 64, 200, 4)
    en_list = [
        torch.randn(3, 64, 200, 79),
        torch.randn(3, 64, 200, 39),
        torch.randn(3, 64, 200, 19),
        torch.randn(3, 64, 200, 9),
        torch.randn(3, 64, 200, 4),
    ]

    net1 = U2Net_Decoder(64, 64, (2, 3), (1, 3), "cat", "iLN")
    net2 = U2Net_Decoder_FrameWise(64, 64, (2, 3), (1, 3), "cat", "iLN")
    net2.load_state_dict(net1.state_dict())
    net2 = torch.jit.script(net2)
    net1.eval()
    net2.eval()

    y2_list = []
    with torch.no_grad():
        y1 = net1(x, en_list)
        for i in range(x.shape[2]):
            en_fw_list = list(map(lambda x: x[:, :, [i]], en_list))
            y2_list += [net2(x[:, :, [i]], en_fw_list)]
    y2 = torch.cat(y2_list, dim=2)
    torch.testing.assert_close(y1, y2)
    ...


if __name__ == "__main__":
    main()
    ...
