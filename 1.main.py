import torch
from tqdm import trange

from network.EaBNet import EaBNet
from network.EaBNet_FrameWise import EaBNet as EaBNetFW
from network.EaBNet_FrameWise_Stateful import EaBNet as EaBNetST


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


def load_state_dict_from1to2(dict1, dict2):
    """dict1 and dict2 have the same value shapes and order."""
    assert len(dict1) == len(dict2)
    for (k1, v1), (k2, v2) in zip(dict1.items(), dict2.items()):
        assert v1.shape == v2.shape
        v2.copy_(v1)
    ...


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
    load_state_dict_from1to2(net1.state_dict(), net2.state_dict())
    net2 = torch.jit.script(net2)
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


def test_stateful():
    kwargs = get_default_config()
    net1 = EaBNet(**kwargs)
    net2 = EaBNetST(**kwargs)
    load_state_dict_from1to2(net1.state_dict(), net2.state_dict())
    net2 = torch.jit.script(net2)
    net1.eval()
    net2.eval()

    M, c, kd1, p, q = kwargs["M"], kwargs["c"], kwargs["kd1"], kwargs["p"], kwargs["q"]
    x = torch.randn(1, 100, 161, M, 2)
    enc_states = [
        torch.zeros(1, 2 * M, 1, 161),
        torch.zeros(1, c, 1, 79),
        torch.zeros(1, c, 1, 39),
        torch.zeros(1, c, 1, 19),
        torch.zeros(1, c, 1, 9),
    ]
    squ_states = [torch.zeros(1, 64, (kd1 - 1) * 2**i, 2) for _ in range(q) for i in range(p)]
    dec_states = [
        torch.zeros(1, 128, 1, 4),
        torch.zeros(1, 128, 1, 9),
        torch.zeros(1, 128, 1, 19),
        torch.zeros(1, 128, 1, 39),
        torch.zeros(1, 128, 1, 79),
    ]
    rnn_states = torch.zeros(2, 161, 64, 2)

    out2_list = []
    with torch.no_grad():
        out1 = net1(x)  # B,2,T,F

        for i in trange(x.shape[1]):
            out, enc_states, squ_states, dec_states, rnn_states = net2(
                x[:, [i]], enc_states, squ_states, dec_states, rnn_states
            )
            out2_list += [out]

    out2 = torch.cat(out2_list, dim=2)

    print(out2.shape, out1.shape)
    torch.testing.assert_close(out2, out1)
    print("Stateful inference test passed!")
    ...


def main():
    # test_model_causality()
    # test_frame_wise()
    test_stateful()
    ...


if __name__ == "__main__":
    main()
    ...
