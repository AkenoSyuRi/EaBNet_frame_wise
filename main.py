import torch
from tqdm import trange

from network.EaBNet import EaBNet
from network.EaBNet_FrameWise import EaBNet as EaBNetFW


def get_default_config():
    k1: tuple = (2, 3)
    k2: tuple = (1, 3)
    c: int = 64
    M: int = 8
    embed_dim: int = 64
    kd1: int = 5
    cd1: int = 64
    d_feat: int = 256
    p: int = 6
    q: int = 3
    is_causal: bool = True
    is_u2: bool = True
    bf_type: str = "lstm"
    topo_type: str = "mimo"
    intra_connect: str = "cat"
    norm_type: str = "iLN"  # note: IN does not support causal inference
    return vars()


def random_init_state_dict(state_dict):
    for key in state_dict.keys():
        state_dict[key].data.uniform_(-0.1, 0.1)


def test_model_causality():
    kwargs = get_default_config()
    nnet = EaBNet(**kwargs)
    nnet.eval()

    x2 = torch.randn(3, 200, 161, 8, 2)
    x1 = x2[:, :100]

    with torch.no_grad():
        out2 = nnet(x2)[:, :, :100]  # B,2,T,F
        out1 = nnet(x1)
    print(out2.shape, out1.shape)
    torch.testing.assert_close(out2, out1)
    print("Causality test passed!")
    ...


def test_frame_wise():
    kwargs = get_default_config()
    net1 = EaBNet(**kwargs)
    net2 = EaBNetFW(**kwargs)
    random_init_state_dict(net1.state_dict())
    net2.load_state_dict(net1.state_dict())
    net1.eval()
    net2.eval()

    x = torch.randn(3, 100, 161, 8, 2)

    out2_list = []
    with torch.no_grad():
        out1 = net1(x)  # B,2,T,F

        for i in trange(x.shape[1]):
            out2_list += [net2(x[:, [i]])]

    out2 = torch.cat(out2_list, dim=2)

    print(out2.shape, out1.shape)
    torch.testing.assert_close(out2, out1)
    print("Frame-wise inference test passed!")
    ...


def main():
    test_model_causality()
    test_frame_wise()
    ...


if __name__ == "__main__":
    main()
    ...
