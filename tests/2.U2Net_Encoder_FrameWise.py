import torch

from network.EaBNet import U2Net_Encoder
from network.EaBNet_FrameWise import U2Net_Encoder as U2Net_Encoder_FrameWise


def main():
    M, k1, k2, c, intra_connect, norm_type = 32, (2, 3), (1, 3), 64, "cat", "iLN"

    net1 = U2Net_Encoder(M * 2, k1, k2, c, intra_connect, norm_type)
    net2 = U2Net_Encoder_FrameWise(M * 2, k1, k2, c, intra_connect, norm_type)
    net2.load_state_dict(net1.state_dict())
    net1.eval()
    net2.eval()

    x = torch.randn(3, 2 * M, 200, 161)
    y2_list = []
    with torch.no_grad():
        y1, _ = net1(x)
        for i in range(x.shape[2]):
            out2, _ = net2(x[:, :, [i]])
            y2_list += [out2]
    y2 = torch.cat(y2_list, dim=2)

    torch.testing.assert_close(y1, y2)
    ...


if __name__ == '__main__':
    main()
    ...
