import os
from pathlib import Path

import openvino as ov
import torch

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
    assert len(dict1) == len(dict2)
    for value in dict2.values():
        value.data.zero_()

    for (k1, v1), (k2, v2) in zip(dict1.items(), dict2.items()):
        assert v1.shape == v2.shape
        v2.copy_(v1)
    ...


def main():
    in_ckpt_path = r"F:\Projects\PycharmProjects\M16_demo\data\ckpt\EaBNet_iLN_epoch67.pth"
    out_onnx_path = Path(r"data/output/EaBNet_iLN_epoch67.onnx")
    kwargs = get_default_config()
    net = EaBNetST(**kwargs)
    state_dict = torch.load(in_ckpt_path, "cpu")
    del state_dict["stft.window"], state_dict["istft.window"]
    load_state_dict_from1to2(state_dict, net.state_dict())

    enc_states = [
        torch.zeros(1, 2 * 8, 1, 161),
        torch.zeros(1, 64, 1, 79),
        torch.zeros(1, 64, 1, 39),
        torch.zeros(1, 64, 1, 19),
        torch.zeros(1, 64, 1, 9),
    ]

    squ_states = [[torch.zeros(1, 64, (5 - 1) * 2**i, 2) for i in range(6)] for _ in range(3)]
    dec_states = [
        torch.zeros(1, 128, 1, 4),
        torch.zeros(1, 128, 1, 9),
        torch.zeros(1, 128, 1, 19),
        torch.zeros(1, 128, 1, 39),
        torch.zeros(1, 128, 1, 79),
    ]
    rnn_state = torch.zeros(2, 161, 64, 2)

    inputs = torch.randn(1, 1, 161, 8, 2)
    n_states = len(enc_states) + len(squ_states) * len(squ_states[0]) + len(dec_states) + 1

    torch.onnx.export(
        net,
        (inputs, enc_states, squ_states, dec_states, rnn_state),
        out_onnx_path.as_posix(),
        input_names=["input", *map(lambda i: f"in_state{i}", range(n_states))],
        output_names=["output", *map(lambda i: f"out_state{i}", range(n_states))],
        opset_version=13,
    )
    print(f"ONNX fp32 model saved to {out_onnx_path}")

    # Simplify the ONNX model using onnx-simplifier
    save_sim_path = out_onnx_path.with_suffix(".sim.onnx")
    os.system(f"onnxsim {out_onnx_path} {save_sim_path}")

    # Convert the ONNX model to OpenVINO IR
    core = ov.Core()
    ov_model = core.read_model(save_sim_path)
    save_vino_path = out_onnx_path.with_suffix(".xml")
    ov.save_model(ov_model, save_vino_path, compress_to_fp16=True)  # enabled by default
    print(f"OpenVINO model saved to {save_vino_path}")

    # # Convert the ONNX model to float16
    # model = onnx.load(save_sim_path.as_posix())
    # model_fp16 = float16.convert_float_to_float16(model)
    # save_fp16_path = out_onnx_path.with_suffix(".sim_fp16.onnx")
    # onnx.save(model_fp16, save_fp16_path.as_posix())
    # print(f"ONNX fp16 model saved to {save_fp16_path}")
    ...


if __name__ == "__main__":
    main()
    ...
