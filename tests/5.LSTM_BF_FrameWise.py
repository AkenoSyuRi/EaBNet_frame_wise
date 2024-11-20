import torch
from torch import nn, Tensor

from network.EaBNet import LSTM_BF


class LSTM_BF_FW(nn.Module):
    def __init__(self, embed_dim: int, M: int, hid_node: int = 64):
        super(LSTM_BF_FW, self).__init__()
        self.embed_dim = embed_dim
        self.M = M
        self.hid_node = hid_node
        self.state = torch.empty(0)
        # self.state1 = None
        # self.state2 = None

        # Components
        self.rnn = nn.LSTM(input_size=embed_dim, hidden_size=hid_node, num_layers=2, batch_first=True)
        # self.rnn2 = nn.LSTM(input_size=hid_node, hidden_size=hid_node, batch_first=True)
        self.w_dnn = nn.Sequential(nn.Linear(hid_node, hid_node), nn.ReLU(True), nn.Linear(hid_node, 2 * M))
        self.norm = nn.LayerNorm([embed_dim])

    def forward(self, embed_x: Tensor) -> Tensor:
        """
        formulate the bf operation
        :param embed_x: (B, C, T, F)
        :return: (B, T, F, M, 2)
        """
        assert embed_x.shape[2] == 1, "The time dimension of input should be 1"
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

        # x, self.state1 = self.rnn1(x, self.state1)
        # x, self.state2 = self.rnn2(x, self.state2)
        x = x.view(B, F, T, -1).transpose(1, 2).contiguous()
        bf_w = self.w_dnn(x).view(B, T, F, self.M, 2)
        return bf_w


def load_state_dict_from1to2(dict1, dict2):
    for k, v in dict1.items():
        if "rnn1" in k:
            k = k.replace("rnn1.", "rnn.")
        if "rnn2" in k:
            k = k.replace("rnn2.", "rnn.").replace("l0", "l1")
        dict2[k].copy_(v)
    ...


def main():
    x = torch.randn(3, 64, 200, 161)
    net1 = LSTM_BF(64, 8)
    net2 = LSTM_BF_FW(64, 8)
    load_state_dict_from1to2(net1.state_dict(), net2.state_dict())
    # net2.load_state_dict(net1.state_dict())
    net2 = torch.jit.script(net2)
    net1.eval()
    net2.eval()

    y2_list = []
    with torch.no_grad():
        y1 = net1(x)
        for i in range(x.shape[2]):
            y2_list += [net2(x[:, :, [i]])]
    y2 = torch.cat(y2_list, dim=1)
    torch.testing.assert_close(y1, y2)
    ...


if __name__ == "__main__":
    main()
    ...
