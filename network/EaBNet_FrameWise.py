from typing import Tuple, Union, List

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.autograd import Variable
from torchaudio import transforms as TT


class EaBNet(nn.Module):
    def __init__(
        self,
        k1: tuple = (2, 3),
        k2: tuple = (1, 3),
        c: int = 64,
        M: int = 9,
        embed_dim: int = 64,
        kd1: int = 5,
        cd1: int = 64,
        d_feat: int = 256,
        p: int = 6,
        q: int = 3,
        is_causal: bool = True,
        is_u2: bool = True,
        bf_type: str = "lstm",
        topo_type: str = "mimo",
        intra_connect: str = "cat",
        norm_type: str = "IN",
    ):
        """
        :param k1: kernel size in the 2-D GLU, (2, 3) by default
        :param k2: kernel size in the UNet-blok, (1, 3) by defauly
        :param c: channel number in the 2-D Convs, 64 by default
        :param M: mic number, 9 by default
        :param embed_dim: embedded dimension, 64 by default
        :param kd1: kernel size in the Squeezed-TCM (dilation-part), 5 by default
        :param cd1: channel number in the Squeezed-TCM (dilation-part), 64 by default
        :param d_feat: channel number in the Squeezed-TCM(pointwise-part), 256 by default
        :param p: the number of Squeezed-TCMs within a group, 6 by default
        :param q: group numbers, 3 by default
        :param is_causal: causal flag, True by default
        :param is_u2: whether U^{2} is set, True by default
        :param bf_type: beamformer type, "lstm" by default
        :param topo_type: topology type, "mimo" and "miso", "mimo" by default
        :param intra_connect: intra connection type, "cat" by default
        :param norm_type: "IN" by default.

        Note: as IN will not accumulate mean and var statistics in both training and inference phase, it can not
        guarantee strict causality. If you wanner use IN, an optional method is to calculate the accumulated statistics
        in both training and inference stages. Besides, you can also choose other norms like BN, LN, cLN.
        """
        super(EaBNet, self).__init__()
        self.k1 = k1
        self.k2 = k2
        self.c = c
        self.M = M
        self.embed_dim = embed_dim
        self.kd1 = kd1
        self.cd1 = cd1
        self.d_feat = d_feat
        self.p = p
        self.q = q
        self.is_causal = is_causal
        self.is_u2 = is_u2
        self.bf_type = bf_type
        self.intra_connect = intra_connect
        self.topo_type = topo_type
        self.norm_type = norm_type

        if is_u2:
            self.en = U2Net_Encoder(M * 2, k1, k2, c, intra_connect, norm_type)
            self.de = U2Net_Decoder(embed_dim, c, k1, k2, intra_connect, norm_type)
        else:
            self.en = UNet_Encoder(M * 2, k1, c, norm_type)
            self.de = UNet_Decoder(embed_dim, k1, c, norm_type)

        if topo_type == "mimo":
            if bf_type == "lstm":
                self.bf_map = LSTM_BF_FW(embed_dim, M)
            elif bf_type == "cnn":
                self.bf_map = nn.Conv2d(embed_dim, M * 2, (1, 1), (1, 1))  # pointwise
        elif topo_type == "miso":
            self.bf_map = nn.Conv2d(embed_dim, 2, (1, 1), (1, 1))  # pointwise

        stcn_list = []
        for _ in range(q):
            stcn_list.append(SqueezedTCNGroup(kd1, cd1, d_feat, p, is_causal, norm_type))
        self.stcns = nn.ModuleList(stcn_list)

    def forward(self, inpt: Tensor) -> Tensor:
        """
        :param inpt: (B, T, F, M, 2) -> (batchsize, seqlen, freqsize, mics, 2)
        :return: beamformed estimation: (B, 2, T, F)
        """
        if inpt.ndim == 4:
            inpt = inpt.unsqueeze(dim=-2)
        b_size, seq_len, freq_len, M, _ = inpt.shape
        x = inpt.transpose(-2, -1).contiguous()
        x = x.view(b_size, seq_len, freq_len, -1).permute(0, 3, 1, 2)
        x, en_list = self.en(x)
        c = x.shape[1]
        x = x.transpose(-2, -1).contiguous().view(b_size, -1, seq_len)
        x_acc = torch.zeros(x.size(), dtype=x.dtype, device=x.device)
        for stcn in self.stcns:
            x = stcn(x)
            x_acc = x_acc + x
        x = x_acc
        x = x.view(b_size, c, -1, seq_len).transpose(-2, -1).contiguous()
        x = self.de(x, en_list)
        if self.topo_type == "mimo":
            if self.bf_type == "lstm":
                bf_w = self.bf_map(x)  # (B, T, F, M, 2)
            elif self.bf_type == "cnn":
                bf_w = self.bf_map(x)
                bf_w = bf_w.view(b_size, M, -1, seq_len, freq_len).permute(0, 3, 4, 1, 2)  # (B,T,F,M,2)
            else:
                raise NotImplementedError("bf_type should be 'lstm' or 'cnn'.")
            bf_w_r, bf_w_i = bf_w[..., 0], bf_w[..., -1]
            esti_x_r, esti_x_i = (bf_w_r * inpt[..., 0] - bf_w_i * inpt[..., -1]).sum(dim=-1), (
                bf_w_r * inpt[..., -1] + bf_w_i * inpt[..., 0]
            ).sum(dim=-1)
            return torch.stack((esti_x_r, esti_x_i), dim=1)
        elif self.topo_type == "miso":
            bf_w = self.bf_map(x)  # (B,2,T,F)
            bf_w = bf_w.permute(0, 2, 3, 1)  # (B,T,F,2)
            bf_w_r, bf_w_i = bf_w[..., 0], bf_w[..., -1]
            # mic-0 is selected as the target mic herein
            esti_x_r, esti_x_i = (bf_w_r * inpt[..., 0, 0] - bf_w_i * inpt[..., 0, -1]), (
                bf_w_r * inpt[..., 0, -1] + bf_w_i * inpt[..., 0, 0]
            )
            return torch.stack((esti_x_r, esti_x_i), dim=1)
        else:
            raise NotImplementedError("topo_type should be 'mimo' or 'miso'.")


class U2Net_Encoder(nn.Module):
    def __init__(
        self,
        cin: int,
        k1: tuple,
        k2: tuple,
        c: int,
        intra_connect: str,
        norm_type: str,
    ):
        super(U2Net_Encoder, self).__init__()
        self.cin = cin
        self.k1 = k1
        self.k2 = k2
        self.c = c
        self.intra_connect = intra_connect
        self.norm_type = norm_type
        k_beg = (2, 5)
        c_end = 64
        meta_unet = []
        meta_unet.append(En_unet_module(cin, c, k_beg, k2, intra_connect, norm_type, scale=4, is_deconv=False))
        meta_unet.append(En_unet_module(c, c, k1, k2, intra_connect, norm_type, scale=3, is_deconv=False))
        meta_unet.append(En_unet_module(c, c, k1, k2, intra_connect, norm_type, scale=2, is_deconv=False))
        meta_unet.append(En_unet_module(c, c, k1, k2, intra_connect, norm_type, scale=1, is_deconv=False))
        self.meta_unet_list = nn.ModuleList(meta_unet)
        self.last_conv = nn.Sequential(
            GateConv2dFW(c, c_end, k1, (1, 2)), NormSwitch(norm_type, "2D", c_end), nn.PReLU(c_end)
        )

    def forward(self, x: Tensor):
        en_list = []
        for meta_unet in self.meta_unet_list:
            x = meta_unet(x)
            en_list.append(x)
        x = self.last_conv(x)
        en_list.append(x)
        return x, en_list


class UNet_Encoder(nn.Module):
    def __init__(
        self,
        cin: int,
        k1: Tuple[int, int],
        c: int,
        norm_type: str,
    ):
        super(UNet_Encoder, self).__init__()
        self.cin = cin
        self.k1 = k1
        self.c = c
        self.norm_type = norm_type
        k_beg = (2, 5)
        c_end = 64
        unet = []
        unet.append(nn.Sequential(GateConv2dFW(cin, c, k_beg, (1, 2)), NormSwitch(norm_type, "2D", c), nn.PReLU(c)))
        unet.append(nn.Sequential(GateConv2dFW(c, c, k1, (1, 2)), nn.PReLU(c)))
        unet.append(nn.Sequential(GateConv2dFW(c, c, k1, (1, 2)), nn.PReLU(c)))
        unet.append(nn.Sequential(GateConv2dFW(c, c, k1, (1, 2)), NormSwitch(norm_type, "2D", c), nn.PReLU(c)))
        unet.append(
            nn.Sequential(GateConv2dFW(c, c_end, k1, (1, 2)), NormSwitch(norm_type, "2D", c_end), nn.PReLU(c_end))
        )
        self.unet_list = nn.ModuleList(unet)

    def forward(self, x: Tensor):
        en_list = []
        for i in range(len(self.unet_list)):
            x = self.unet_list[i](x)
            en_list.append(x)
        return x, en_list


class U2Net_Decoder(nn.Module):
    def __init__(self, embed_dim, c, k1, k2, intra_connect, norm_type):
        super(U2Net_Decoder, self).__init__()
        self.embed_dim = embed_dim
        self.k1 = k1
        self.k2 = k2
        self.c = c
        self.intra_connect = intra_connect
        self.norm_type = norm_type
        c_beg = 64
        k_end = (2, 5)

        meta_unet = []
        meta_unet.append(En_unet_module(c_beg * 2, c, k1, k2, intra_connect, norm_type, scale=1, is_deconv=True))
        meta_unet.append(En_unet_module(c * 2, c, k1, k2, intra_connect, norm_type, scale=2, is_deconv=True))
        meta_unet.append(En_unet_module(c * 2, c, k1, k2, intra_connect, norm_type, scale=3, is_deconv=True))
        meta_unet.append(En_unet_module(c * 2, c, k1, k2, intra_connect, norm_type, scale=4, is_deconv=True))
        self.meta_unet_list = nn.ModuleList(meta_unet)
        self.last_conv = nn.Sequential(
            GateConvTranspose2dFW(c * 2, embed_dim, k_end, (1, 2)),
            NormSwitch(norm_type, "2D", embed_dim),
            nn.PReLU(embed_dim),
        )

    def forward(self, x: Tensor, en_list: List[Tensor]) -> Tensor:
        for i, meta_unet in enumerate(self.meta_unet_list):
            tmp = torch.cat((x, en_list[-(i + 1)]), dim=1)
            x = meta_unet(tmp)
        x = torch.cat((x, en_list[0]), dim=1)
        x = self.last_conv(x)
        return x


class UNet_Decoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        k1: tuple,
        c: int,
        norm_type: str,
    ):
        super(UNet_Decoder, self).__init__()
        self.embed_dim = embed_dim
        self.k1 = k1
        self.c = c
        self.norm_type = norm_type
        c_beg = 64  # the channels of the last encoder and the first decoder are fixed at 64 by default
        k_end = (2, 5)
        unet = []
        unet.append(
            nn.Sequential(GateConvTranspose2dFW(c_beg * 2, c, k1, (1, 2)), NormSwitch(norm_type, "2D", c), nn.PReLU(c))
        )
        unet.append(
            nn.Sequential(GateConvTranspose2dFW(c * 2, c, k1, (1, 2)), NormSwitch(norm_type, "2D", c), nn.PReLU(c))
        )
        unet.append(
            nn.Sequential(GateConvTranspose2dFW(c * 2, c, k1, (1, 2)), NormSwitch(norm_type, "2D", c), nn.PReLU(c))
        )
        unet.append(
            nn.Sequential(GateConvTranspose2dFW(c * 2, c, k1, (1, 2)), NormSwitch(norm_type, "2D", c), nn.PReLU(c))
        )
        unet.append(
            nn.Sequential(
                GateConvTranspose2dFW(c * 2, embed_dim, k_end, (1, 2)),
                NormSwitch(norm_type, "2D", embed_dim),
                nn.PReLU(embed_dim),
            )
        )
        self.unet_list = nn.ModuleList(unet)

    def forward(self, x: Tensor, en_list: list) -> Tensor:
        for i in range(len(self.unet_list)):
            tmp = torch.cat((x, en_list[-(i + 1)]), dim=1)  # skip connections
            x = self.unet_list[i](tmp)
        return x


class En_unet_module(nn.Module):
    def __init__(
        self,
        cin: int,
        cout: int,
        k1: tuple,
        k2: tuple,
        intra_connect: str,
        norm_type: str,
        scale: int,
        is_deconv: bool,
    ):
        super(En_unet_module, self).__init__()
        self.k1 = k1
        self.k2 = k2
        self.cin = cin
        self.cout = cout
        self.intra_connect = intra_connect
        self.scale = scale
        self.is_deconv = is_deconv

        in_conv_list = []
        if not is_deconv:
            in_conv_list.append(GateConv2dFW(cin, cout, k1, (1, 2)))
        else:
            in_conv_list.append(GateConvTranspose2dFW(cin, cout, k1, (1, 2)))
        in_conv_list.append(NormSwitch(norm_type, "2D", cout))
        in_conv_list.append(nn.PReLU(cout))
        self.in_conv = nn.Sequential(*in_conv_list)

        enco_list, deco_list = [], []
        for _ in range(scale):
            enco_list.append(Conv2dunit(k2, cout, norm_type))
        for i in range(scale):
            if i == 0:
                deco_list.append(Deconv2dunit(k2, cout, "add", norm_type))
            else:
                deco_list.append(Deconv2dunit(k2, cout, intra_connect, norm_type))
        self.enco = nn.ModuleList(enco_list)
        self.deco = nn.ModuleList(deco_list)
        self.skip_connect = Skip_connect(intra_connect)

    def forward(self, x):
        x_resi = self.in_conv(x)
        x = x_resi
        x_list = []
        for enco_layer in self.enco:
            x = enco_layer(x)
            x_list.append(x)

        for i, deco_layer in enumerate(self.deco):
            if i == 0:
                x = deco_layer(x)
            else:
                x_con = self.skip_connect(x, x_list[-(i + 1)])
                x = deco_layer(x_con)
        x_resi = x_resi + x
        del x_list
        return x_resi


class Conv2dunit(nn.Module):
    def __init__(
        self,
        k: tuple,
        c: int,
        norm_type: str,
    ):
        super(Conv2dunit, self).__init__()
        self.k = k
        self.c = c
        self.norm_type = norm_type
        self.conv = nn.Sequential(nn.Conv2d(c, c, k, (1, 2)), NormSwitch(norm_type, "2D", c), nn.PReLU(c))

    def forward(self, x):
        return self.conv(x)


class Deconv2dunit(nn.Module):
    def __init__(
        self,
        k: tuple,
        c: int,
        intra_connect: str,
        norm_type: str,
    ):
        super(Deconv2dunit, self).__init__()
        self.k, self.c = k, c
        self.intra_connect = intra_connect
        self.norm_type = norm_type
        deconv_list = []
        if self.intra_connect == "add":
            deconv_list.append(nn.ConvTranspose2d(c, c, k, (1, 2)))
        elif self.intra_connect == "cat":
            deconv_list.append(nn.ConvTranspose2d(2 * c, c, k, (1, 2)))
        deconv_list.append(NormSwitch(norm_type, "2D", c)),
        deconv_list.append(nn.PReLU(c))
        self.deconv = nn.Sequential(*deconv_list)

    def forward(self, x: Tensor) -> Tensor:
        return self.deconv(x)


class GateConv2dFW(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int],
        stride: Tuple[int, int],
    ):
        super(GateConv2dFW, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        assert stride[0] == 1, f"{self.__class__.__name__} only supports stride[0] == 1"
        self.state = torch.empty(0)

        k_t = kernel_size[0]
        if k_t > 1:
            self.conv = nn.Sequential(
                nn.Identity(),
                nn.Conv2d(
                    in_channels=in_channels, out_channels=out_channels * 2, kernel_size=kernel_size, stride=stride
                ),
            )
        else:
            self.conv = nn.Conv2d(
                in_channels=in_channels, out_channels=out_channels * 2, kernel_size=kernel_size, stride=stride
            )

    def forward(self, inputs: Tensor) -> Tensor:
        """inputs: (batch_size, channels, time=1, freq)"""
        # assert inputs.shape[-2] == 1, f"{self.__class__.__name__} only supports single frame input"

        if inputs.ndim == 3:
            inputs = inputs.unsqueeze(dim=1)

        if self.state.shape[0] == 0:
            B, C, _, F = inputs.shape
            self.state = torch.zeros(B, C, self.kernel_size[0] - 1, F, dtype=inputs.dtype, device=inputs.device)

        inputs = torch.cat([self.state, inputs], dim=2)
        self.state = inputs[:, :, 1:]

        x = self.conv(inputs)
        outputs, gate = x.chunk(2, dim=1)
        return outputs * gate.sigmoid()


class GateConvTranspose2dFW(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int],
        stride: Tuple[int, int],
    ):
        super(GateConvTranspose2dFW, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        assert stride[0] == 1, f"{self.__class__.__name__} only supports stride[0] == 1"
        self.state = torch.empty(0)

        k_t = kernel_size[0]
        if k_t > 1:
            self.conv = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=in_channels,
                    out_channels=out_channels * 2,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=(k_t - 1, 0),
                ),
                nn.Identity(),
            )
        else:
            self.conv = nn.ConvTranspose2d(
                in_channels=in_channels, out_channels=out_channels * 2, kernel_size=kernel_size, stride=stride
            )

    def forward(self, inputs: Tensor) -> Tensor:
        """inputs: (batch_size, channels, time=1, freq)"""
        # assert inputs.shape[-2] == 1, f"{self.__class__.__name__} only supports single frame input"

        if inputs.ndim == 3:
            inputs = inputs.unsqueeze(dim=1)

        if self.state.shape[0] == 0:
            B, C, _, F = inputs.shape
            self.state = torch.zeros(B, C, self.kernel_size[0] - 1, F, dtype=inputs.dtype, device=inputs.device)

        inputs = torch.cat([self.state, inputs], dim=2)
        self.state = inputs[:, :, 1:]

        x = self.conv(inputs)
        outputs, gate = x.chunk(2, dim=1)
        return outputs * gate.sigmoid()


class Skip_connect(nn.Module):
    def __init__(self, connect):
        super(Skip_connect, self).__init__()
        self.connect = connect

    def forward(self, x_main, x_aux):
        if self.connect == "add":
            x = x_main + x_aux
        elif self.connect == "cat":
            x = torch.cat((x_main, x_aux), dim=1)
        else:
            raise ValueError(f"Unsupported intra_connect type: {self.connect}")
        return x


class SqueezedTCNGroup(nn.Module):
    def __init__(
        self,
        kd1: int,
        cd1: int,
        d_feat: int,
        p: int,
        is_causal: bool,
        norm_type: str,
    ):
        super(SqueezedTCNGroup, self).__init__()
        self.kd1 = kd1
        self.cd1 = cd1
        self.d_feat = d_feat
        self.p = p
        self.is_causal = is_causal
        self.norm_type = norm_type

        # Components
        self.tcm_list = nn.ModuleList([SqueezedTCM_FW(kd1, cd1, 2**i, d_feat, is_causal, norm_type) for i in range(p)])

    def forward(self, x):
        for tcm in self.tcm_list:
            x = tcm(x)
        return x


class CausalConstantPad1d(nn.Module):
    def __init__(self, padding, value=0.0):
        super(CausalConstantPad1d, self).__init__()
        self.padding = padding
        self.value = value
        self.state = torch.empty(0)

    def forward(self, inputs: Tensor):
        # assert inputs.shape[-1] == 1, "CausalConstantPad1d only supports 1 time frame input"

        if self.state.shape[0] == 0:
            inputs = nn.functional.pad(inputs, (*self.padding, 0, 0), "constant", self.value)
        else:
            inputs = torch.cat([self.state, inputs], dim=-1)
        self.state = inputs[..., 1:]
        return inputs


class SqueezedTCM_FW(nn.Module):
    def __init__(
        self,
        kd1: int,
        cd1: int,
        dilation: int,
        d_feat: int,
        is_causal: bool,
        norm_type: str,
    ):
        super(SqueezedTCM_FW, self).__init__()
        self.kd1 = kd1
        self.cd1 = cd1
        self.dilation = dilation
        self.d_feat = d_feat
        self.is_causal = is_causal
        self.norm_type = norm_type

        self.in_conv = nn.Conv1d(d_feat, cd1, 1, bias=False)
        if is_causal:
            self.pad = ((kd1 - 1) * dilation, 0)
        else:
            self.pad = ((kd1 - 1) * dilation // 2, (kd1 - 1) * dilation // 2)
        self.left_conv = nn.Sequential(
            nn.PReLU(cd1),
            NormSwitch(norm_type, "1D", cd1),
            CausalConstantPad1d(self.pad),
            nn.Conv1d(cd1, cd1, kd1, dilation=dilation, bias=False),
        )
        self.right_conv = nn.Sequential(
            nn.PReLU(cd1),
            NormSwitch(norm_type, "1D", cd1),
            CausalConstantPad1d(self.pad),
            nn.Conv1d(cd1, cd1, kernel_size=kd1, dilation=dilation, bias=False),
            nn.Sigmoid(),
        )
        self.out_conv = nn.Sequential(
            nn.PReLU(cd1), NormSwitch(norm_type, "1D", cd1), nn.Conv1d(cd1, d_feat, kernel_size=1, bias=False)
        )

    def forward(self, x):
        resi = x
        x = self.in_conv(x)
        x = self.left_conv(x) * self.right_conv(x)
        x = self.out_conv(x)
        x = x + resi
        return x


class LSTM_BF_FW(nn.Module):
    def __init__(self, embed_dim: int, M: int, hid_node: int = 64):
        super(LSTM_BF_FW, self).__init__()
        self.embed_dim = embed_dim
        self.M = M
        self.hid_node = hid_node
        self.state = torch.empty(0)

        # Components
        self.rnn = nn.LSTM(input_size=embed_dim, hidden_size=hid_node, num_layers=2, batch_first=True)
        self.w_dnn = nn.Sequential(nn.Linear(hid_node, hid_node), nn.ReLU(True), nn.Linear(hid_node, 2 * M))
        self.norm = nn.LayerNorm([embed_dim])

    def forward(self, embed_x: Tensor) -> Tensor:
        """
        formulate the bf operation
        :param embed_x: (B, C, T, F)
        :return: (B, T, F, M, 2)
        """
        # assert embed_x.shape[2] == 1, "The time dimension of input should be 1"
        # norm
        B, _, T, F = embed_x.shape
        x = self.norm(embed_x.permute(0, 3, 2, 1).contiguous())
        x = x.view(B * F, T, -1)

        if self.state.shape[0] == 0:
            x, state = self.rnn(x)
        else:
            state = self.state[..., 0], self.state[..., 1]
            x, state = self.rnn(x, state)
        self.state = torch.stack(state, dim=-1)

        x = x.view(B, F, T, -1).transpose(1, 2).contiguous()
        bf_w = self.w_dnn(x).view(B, T, F, self.M, 2)
        return bf_w


class Chomp_T(nn.Module):
    def __init__(self, t):
        super(Chomp_T, self).__init__()
        self.t = t

    def forward(self, x):
        return x[:, :, : -self.t, :]


class NormSwitch(nn.Module):
    """
    Currently, BN, IN, and cLN are considered
    """

    def __init__(
        self,
        norm_type: str,
        dim_size: str,
        c: int,
    ):
        super(NormSwitch, self).__init__()
        self.norm_type = norm_type
        self.dim_size = dim_size
        self.c = c

        assert norm_type in ["BN", "IN", "cLN", "iLN"] and dim_size in ["1D", "2D"]
        if norm_type == "BN":
            if dim_size == "1D":
                self.norm = nn.BatchNorm1d(c)
            else:
                self.norm = nn.BatchNorm2d(c)
        elif norm_type == "IN":
            if dim_size == "1D":
                self.norm = nn.InstanceNorm1d(c, affine=True)
            else:
                self.norm = nn.InstanceNorm2d(c, affine=True)
        elif norm_type == "cLN":
            if dim_size == "1D":
                self.norm = CumulativeLayerNorm1d(c, affine=True)
            else:
                self.norm = CumulativeLayerNorm2d(c, affine=True)
        elif norm_type == "iLN":
            if dim_size == "1D":
                self.norm = InstantLayerNorm1d(c, affine=True)
            else:
                self.norm = InstantLayerNorm2d(c, affine=True)

    def forward(self, x):
        return self.norm(x)


class InstantLayerNorm1d(nn.Module):
    def __init__(
        self,
        num_features,
        affine=True,
        eps=1e-5,
    ):
        super(InstantLayerNorm1d, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if affine:
            self.gain = nn.Parameter(torch.ones(1, num_features, 1), requires_grad=True)
            self.bias = nn.Parameter(torch.zeros(1, num_features, 1), requires_grad=True)
        else:
            self.gain = Variable(torch.ones(1, num_features, 1), requires_grad=False)
            self.bias = Variable(torch.zeros(1, num_features, 1), requires_gra=False)

    def forward(self, inpt):
        # inpt: (B,C,T)
        # b_size, channel, seq_len = inpt.shape
        ins_mean = torch.mean(inpt, dim=1, keepdim=True)  # (B,1,T)
        ins_std = (torch.var(inpt, dim=1, keepdim=True) + self.eps).pow(0.5)  # (B,1,T)
        x = (inpt - ins_mean) / ins_std
        return x * self.gain.expand_as(x).type(x.type()) + self.bias.expand_as(x).type(x.type())


class InstantLayerNorm2d(nn.Module):
    def __init__(
        self,
        num_features,
        affine=True,
        eps=1e-5,
    ):
        super(InstantLayerNorm2d, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps
        if affine:
            self.gain = nn.Parameter(torch.ones(1, num_features, 1, 1), requires_grad=True)
            self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1), requires_grad=True)
        else:
            self.gain = Variable(torch.ones(1, num_features, 1, 1), requires_grad=False)
            self.bias = Variable(torch.zeros(1, num_features, 1, 1), requires_grad=False)

    def forward(self, inpt):
        # inpt: (B,C,T,F)
        ins_mean = torch.mean(inpt, dim=[1, 3], keepdim=True)  # (B,C,T,1)
        ins_std = (torch.std(inpt, dim=[1, 3], keepdim=True) + self.eps).pow(0.5)  # (B,C,T,1)
        x = (inpt - ins_mean) / ins_std
        return x * self.gain.expand_as(x).type(x.type()) + self.bias.expand_as(x).type(x.type())


class CumulativeLayerNorm1d(nn.Module):
    def __init__(
        self,
        num_features,
        affine=True,
        eps=1e-5,
    ):
        super(CumulativeLayerNorm1d, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if affine:
            self.gain = nn.Parameter(torch.ones(1, num_features, 1), requires_grad=True)
            self.bias = nn.Parameter(torch.zeros(1, num_features, 1), requires_grad=True)
        else:
            self.gain = Variable(torch.ones(1, num_features, 1), requires_grad=False)
            self.bias = Variable(torch.zeros(1, num_features, 1), requires_gra=False)

    def forward(self, inpt):
        # inpt: (B,C,T)
        b_size, channel, seq_len = inpt.shape
        cum_sum = torch.cumsum(inpt.sum(1), dim=1)  # (B,T)
        cum_power_sum = torch.cumsum(inpt.pow(2).sum(1), dim=1)  # (B,T)

        entry_cnt = torch.arange(channel, channel * (seq_len + 1), channel)
        # entry_cnt = np.arange(channel, channel * (seq_len + 1), channel)
        # entry_cnt = torch.from_numpy(entry_cnt).type(inpt.type())
        entry_cnt = entry_cnt.view(1, -1).expand_as(cum_sum)  # (B,T)

        cum_mean = cum_sum / entry_cnt  # (B,T)
        cum_var = (cum_power_sum - 2 * cum_mean * cum_sum) / entry_cnt + cum_mean.pow(2)
        cum_std = (cum_var + self.eps).sqrt()

        x = (inpt - cum_mean.unsqueeze(dim=1).expand_as(inpt)) / cum_std.unsqueeze(dim=1).expand_as(inpt)
        return x * self.gain.expand_as(x).type(x.type()) + self.bias.expand_as(x).type(x.type())


class CumulativeLayerNorm2d(nn.Module):
    def __init__(
        self,
        num_features,
        affine=True,
        eps=1e-5,
    ):
        super(CumulativeLayerNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine

        if affine:
            self.gain = nn.Parameter(torch.ones(1, num_features, 1, 1))
            self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        else:
            self.gain = Variable(torch.ones(1, num_features, 1, 1), requires_grad=False)
            self.bias = Variable(torch.zeros(1, num_features, 1, 1), requires_grad=False)

    def forward(self, inpt):
        """
        :param inpt: (B,C,T,F)
        :return:
        """
        b_size, channel, seq_len, freq_num = inpt.shape
        step_sum = inpt.sum([1, 3], keepdim=True)  # (B,1,T,1)
        step_pow_sum = inpt.pow(2).sum([1, 3], keepdim=True)  # (B,1,T,1)
        cum_sum = torch.cumsum(step_sum, dim=-2)  # (B,1,T,1)
        cum_pow_sum = torch.cumsum(step_pow_sum, dim=-2)  # (B,1,T,1)

        entry_cnt = torch.arange(channel * freq_num, channel * freq_num * (seq_len + 1), channel * freq_num)
        # entry_cnt = np.arange(channel * freq_num, channel * freq_num * (seq_len + 1), channel * freq_num)
        # entry_cnt = torch.from_numpy(entry_cnt).type(inpt.type())
        entry_cnt = entry_cnt.view(1, 1, seq_len, 1).expand_as(cum_sum)

        cum_mean = cum_sum / entry_cnt
        cum_var = (cum_pow_sum - 2 * cum_mean * cum_sum) / entry_cnt + cum_mean.pow(2)
        cum_std = (cum_var + self.eps).sqrt()

        x = (inpt - cum_mean) / cum_std
        return x * self.gain.expand_as(x).type(x.type()) + self.bias.expand_as(x).type(x.type())


def com_mag_mse_loss(preds: Tensor, target: Tensor):
    """preds, target: (B,2,T,F)"""
    esti, label = preds, target
    B, _, T, F = esti.shape
    with torch.no_grad():
        mask_for_loss = torch.ones(B, T, F, device=esti.device, dtype=esti.dtype)
        com_mask_for_loss = torch.stack((mask_for_loss, mask_for_loss), dim=1)  # B,2,T,F
    mag_esti, mag_label = torch.norm(esti, dim=1), torch.norm(label, dim=1)
    loss1 = (((mag_esti - mag_label) ** 2.0) * mask_for_loss).sum() / mask_for_loss.sum()
    loss2 = (((esti - label) ** 2.0) * com_mask_for_loss).sum() / com_mask_for_loss.sum()
    return 0.5 * (loss1 + loss2)


def numParams(net):
    import numpy as np

    num = 0
    for param in net.parameters():
        if param.requires_grad:
            num += int(np.prod(param.size()))
    return num / 1e6


class EaBNetHelper(nn.Module):
    def __init__(self, win_len=320, hop_len=160, comp_factor=0.5, **kwargs):
        super().__init__()
        self.comp_factor = comp_factor

        self.stft = TT.Spectrogram(n_fft=win_len, win_length=win_len, hop_length=hop_len, power=None)
        self.istft = TT.InverseSpectrogram(n_fft=win_len, win_length=win_len, hop_length=hop_len)
        self.net = EaBNet(**kwargs)

    def apply_stft(self, in_data: Tensor):
        """
        in_data: (B,C,L) or (B,L)
        return: (B,T,F,C,2) or (B,2,T,F)
        """
        is_mono = in_data.ndim == 2
        if is_mono:
            in_data = torch.unsqueeze(in_data, dim=1)  # B,1,L
        in_spec = self.stft(in_data)  # B,C,F,T

        x = torch.permute(in_spec, [0, 3, 2, 1])  # B,T,F,C
        comp_mag, phase = torch.abs(x) ** self.comp_factor, torch.angle(x)
        x = torch.view_as_real(comp_mag * torch.exp(1j * phase))  # B,T,F,C,2

        if is_mono:
            x = torch.squeeze(x, dim=-2)  # B,T,F,2
            x = torch.permute(x, [0, 3, 1, 2])  # B,2,T,F
        return x

    def apply_istft(self, est_spec: Tensor):
        """
        est_spec: (B,2,T,F)
        return: (B,L)
        """
        com_mag, phase = torch.norm(est_spec, dim=1), torch.atan2(est_spec[:, 1], est_spec[:, 0])
        ehc_spec = com_mag ** (1 / self.comp_factor) * torch.exp(1j * phase)

        ehc_spec = torch.permute(ehc_spec, [0, 2, 1])  # B,F,T
        out_data = self.istft(ehc_spec)
        return out_data

    def forward(self, in_data: Tensor, target: Tensor = None) -> Union[Tensor, Tuple[Tensor, Tensor, Tensor]]:
        """
        in_data: (B,C,L)
        target: (B,L)
        return: (B,L)
        """
        x = self.apply_stft(in_data)  # B,T,F,C,2
        est_spec = self.net(x)  # B,2,T,F
        out_data = self.apply_istft(est_spec)

        if target is not None:  # for loss calculation
            tar_spec = self.apply_stft(target)  # B,2,T,F
            return out_data, est_spec, tar_spec

        return out_data


def main(args, net, device):
    batch_size = args.batch_size
    mics = args.mics
    sr = args.sr
    wav_len = int(args.wav_len * sr)

    in_data = torch.randn(batch_size, mics, wav_len).to(device)

    est_data = net(in_data)
    print(f"est_data: {est_data.shape}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        "This script provides the network code and a simple testing, you can train the"
        "network according to your own pipeline"
    )
    parser.add_argument("--batch_size", type=int, default=3)
    parser.add_argument("--mics", type=int, default=8)
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--wav_len", type=float, default=4.0)
    parser.add_argument("--win_size", type=float, default=0.020)
    parser.add_argument("--win_shift", type=float, default=0.010)
    parser.add_argument("--fft_num", type=int, default=320)
    parser.add_argument("--comp_factor", type=float, default=0.5)
    parser.add_argument("--k1", type=tuple, default=(2, 3))
    parser.add_argument("--k2", type=tuple, default=(1, 3))
    parser.add_argument("--c", type=int, default=64)
    parser.add_argument("--M", type=int, default=8)
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--kd1", type=int, default=5)
    parser.add_argument("--cd1", type=int, default=64)
    parser.add_argument("--d_feat", type=int, default=256)
    parser.add_argument("--p", type=int, default=6)
    parser.add_argument("--q", type=int, default=3)
    parser.add_argument("--is_causal", type=bool, default=True, choices=[True, False])
    parser.add_argument("--is_u2", type=bool, default=True, choices=[True, False])
    parser.add_argument("--bf_type", type=str, default="lstm", choices=["lstm", "cnn"])
    parser.add_argument("--topo_type", type=str, default="mimo", choices=["mimo", "miso"])
    parser.add_argument("--intra_connect", type=str, default="cat", choices=["cat", "add"])
    parser.add_argument("--norm_type", type=str, default="iLN", choices=["BN", "IN", "cLN", "iLN"])
    args = parser.parse_args()

    device = "cpu"
    net = EaBNetHelper(
        win_len=int(args.win_size * args.sr),
        hop_len=int(args.win_shift * args.sr),
        comp_factor=args.comp_factor,
        k1=args.k1,
        k2=args.k2,
        c=args.c,
        M=args.M,
        embed_dim=args.embed_dim,
        kd1=args.kd1,
        cd1=args.cd1,
        d_feat=args.d_feat,
        p=args.p,
        q=args.q,
        is_causal=args.is_causal,
        is_u2=args.is_u2,
        bf_type=args.bf_type,
        topo_type=args.topo_type,
        intra_connect=args.intra_connect,
        norm_type=args.norm_type,
    ).to(device)
    net.eval()
    print("The number of trainable parameters is: {}M".format(numParams(net)))
    # from ptflops.flops_counter import get_model_complexity_info

    # get_model_complexity_info(net, (7, 96000))
    main(args, net, device)

    import torchinfo

    torchinfo.summary(net, input_size=(3, 8, 64000), col_names=("input_size", "output_size", "num_params"))
    ...
